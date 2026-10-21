"""`sdpb`-compatible command line (SDPB 3.1.0: sdpb/SDPB_Parameters.cxx, sdpb/solve.cxx).

    sdpvibe -s SDP [--precision P] [-o OUT] [-c CK] [-i INITIAL_CK] [--writeSolution x,y,z,X,Y] ...

Same options, defaults, files and terminate reasons as `sdpb`: OUT defaults to SDP + "_out", CK to SDP + ".ck";
a checkpoint in the initial checkpoint directory (default CK) is loaded if present (required if -i is given);
checkpoints are written every --checkpointInterval seconds and at the end (unless --noFinalCheckpoint);
SIGTERM stops at the start of the next iteration, saves a checkpoint and the solution, and exits with code 15.
Options that only concern SDPB's MPI parallelisation (--procsPerNode, --procGranularity, --maxSharedMemory,
--maxMemory) are accepted and ignored.  Extension: --gpus N | i,j,... | all (multi-GPU; default: env
SDPVIBE_GPUS, else one GPU).  GPU memory: see README (SDPB_GPU_MEM_GB etc.).
"""
import datetime, os, signal, sys, time
import mpmath as mp

VERSION = "0.2.0"
SDPB_VERSION = "3.1.0"

# name -> (default, kind)   kind: str | int | float | bool | path
OPTIONS = {
    "sdpDir": (None, "path"), "outDir": (None, "path"), "paramFile": (None, "path"),
    "checkpointDir": (None, "path"), "initialCheckpointDir": (None, "path"),
    "noFinalCheckpoint": (False, "bool"), "writeSolution": ("x,y", "str"), "verbosity": ("1", "str"),
    "procsPerNode": (None, "str"), "procGranularity": ("1", "str"), "maxSharedMemory": ("0", "str"), "maxMemory": ("0", "str"),
    "gpus": (None, "str"),                     # SDPviBe: GPUs to use, "2" / "0,1" / "all" (default SDPVIBE_GPUS, else 1)
    "precision": ("400", "int"),
    "findPrimalFeasible": (False, "bool"), "findDualFeasible": (False, "bool"),
    "detectPrimalFeasibleJump": (False, "bool"), "detectDualFeasibleJump": (False, "bool"),
    "maxIterations": ("500", "int"), "maxRuntime": ("9223372036854775807", "int"), "checkpointInterval": ("3600", "int"),
    "dualityGapThreshold": ("1e-30", "num"), "primalErrorThreshold": ("1e-30", "num"), "dualErrorThreshold": ("1e-30", "num"),
    "initialMatrixScalePrimal": ("1e+20", "num"), "initialMatrixScaleDual": ("1e+20", "num"),
    "feasibleCenteringParameter": ("0.1", "num"), "infeasibleCenteringParameter": ("0.3", "num"),
    "stepLengthReduction": ("0.7", "num"), "minPrimalStep": ("0", "num"), "minDualStep": ("0", "num"),
    "maxComplementarity": ("1e+100", "num"),
}
SHORT = {"s": "sdpDir", "o": "outDir", "c": "checkpointDir", "i": "initialCheckpointDir", "p": "paramFile"}
VERBOSITY = {"0": 0, "none": 0, "1": 1, "regular": 1, "2": 2, "debug": 2, "3": 3, "trace": 3}


class UsageError(Exception):
    pass


def parse_args(argv):
    given = {}
    flags = {"help": False, "version": False}
    i = 0
    while i < len(argv):
        a = argv[i]; i += 1
        if a in ("-h", "--help"): flags["help"] = True; continue
        if a == "--version": flags["version"] = True; continue
        if a.startswith("--"):
            name, eq, val = a[2:].partition("=")
        elif a.startswith("-") and len(a) >= 2 and a[1] in SHORT:
            name, val, eq = SHORT[a[1]], a[2:], "=" if len(a) > 2 else ""
        else:
            raise UsageError(f"unrecognised argument '{a}'")
        if name not in OPTIONS:
            raise UsageError(f"unrecognised option '--{name}'")
        if OPTIONS[name][1] == "bool":
            given[name] = True if not eq else val.lower() in ("1", "true", "yes", "on")
            continue
        if not eq:
            if i >= len(argv):
                raise UsageError(f"the required argument for option '--{name}' is missing")
            val = argv[i]; i += 1
        given[name] = val
    params = {k: v[0] for k, v in OPTIONS.items()}
    if "paramFile" in given:                                    # key=value lines; the command line wins
        for line in open(given["paramFile"]):
            line = line.split("#", 1)[0].strip()
            if not line: continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k not in OPTIONS:
                raise UsageError(f"unrecognised option '{k}' in {given['paramFile']}")
            params[k] = (v.lower() in ("1", "true", "yes", "on")) if OPTIONS[k][1] == "bool" else v
    params.update(given)
    return params, flags, set(given)


def resolve(params, given):
    if params["sdpDir"] is None:
        raise UsageError("the option '--sdpDir' is required but missing")
    sdp = os.path.normpath(params["sdpDir"])
    base = sdp if os.path.basename(sdp) != "." else os.path.dirname(sdp)
    if params["outDir"] is None: params["outDir"] = base + "_out"
    if params["checkpointDir"] is None: params["checkpointDir"] = base + ".ck"
    params["requireInitialCheckpoint"] = "initialCheckpointDir" in given
    if params["initialCheckpointDir"] is None: params["initialCheckpointDir"] = params["checkpointDir"]
    v = str(params["verbosity"]).lower()
    if v not in VERBOSITY: raise UsageError(f"invalid verbosity '{params['verbosity']}'")
    params["verbosity"] = VERBOSITY[v]
    parts = [s.strip() for s in params["writeSolution"].split(",") if s.strip()]
    for s in parts:
        if s not in ("x", "y", "z", "X", "Y"):
            raise UsageError(f"invalid --writeSolution element '{s}' (allowed: x, y, z, X, Y)")
    params["writeSolutionParts"] = set(parts)
    for k in ("precision", "maxIterations", "maxRuntime", "checkpointInterval"):
        params[k] = int(params[k])
    return params


def solver_params(params):
    from .solver import DEFAULTS
    out = {}
    for k in DEFAULTS:
        out[k] = params[k] if OPTIONS[k][1] in ("bool", "int") else str(params[k])
    return out


def print_parameters(params):
    order = ["maxIterations", "maxRuntime", "checkpointInterval", "maxSharedMemory", "findPrimalFeasible",
             "findDualFeasible", "detectPrimalFeasibleJump", "detectDualFeasibleJump", "precision",
             "dualityGapThreshold", "primalErrorThreshold", "dualErrorThreshold", "initialMatrixScalePrimal",
             "initialMatrixScaleDual", "feasibleCenteringParameter", "infeasibleCenteringParameter",
             "stepLengthReduction", "maxComplementarity", "initialCheckpointDir", "checkpointDir",
             "noFinalCheckpoint", "writeSolution", "procGranularity", "verbosity"]
    print("Parameters:")
    for k in order:
        v = params[k]
        if isinstance(v, bool): v = "true" if v else "false"
        elif k in ("initialCheckpointDir", "checkpointDir"): v = f'"{os.path.abspath(v)}"'
        elif k == "precision": v = f"{v}({v})"
        name = "precision(actual)" if k == "precision" else k
        print(f"{name:<29}= {v}")
    print()


def iteration_line(row, mu_f, steps):
    f = lambda k: float(mp.mpf(row[k]))
    P_step, D_step, beta = (float(x) for x in steps)
    return (f"{row['iteration']:<4}  {int(row['total_time']):>8} {float(mu_f):<#8.2g} {f('P-obj'):<+#11.3g} "
            f"{f('D-obj'):<+#11.3g} {f('gap'):<#10.3g} {f('P-err'):<+#11.3g} {f('p-err'):<+#11.3g} "
            f"{f('D-err'):<+#11.3g} {P_step:<#8.3g} {D_step:<#8.3g} {beta:<#4.3g}")


def options_record(params):
    return {k: (("true" if params[k] else "false") if isinstance(params[k], bool) else str(params[k]))
            for k in OPTIONS if k not in ("paramFile",) and params.get(k) is not None}


def main(argv=None):
    start = time.time()
    argv = sys.argv[1:] if argv is None else argv
    try:
        params, flags, given = parse_args(argv)
        if flags["help"]:
            print(__doc__); print("Options (as sdpb " + SDPB_VERSION + "):")
            for k, (d, kind) in OPTIONS.items():
                print(f"  --{k}" + ("" if kind == "bool" else f" arg (={d})" if d is not None else " arg"))
            return 0
        if flags["version"]:
            print(f"SDPviBe {VERSION} (drop-in for SDPB {SDPB_VERSION})"); return 0
        params = resolve(params, given)
    except (UsageError, OSError) as e:
        print(f"Error: {e}\nRun with --help for the list of options.", file=sys.stderr)
        return 1
    verb = params["verbosity"]
    out_dir = params["outDir"]

    from . import dist as D
    from .solver import resolve_devices
    if not D.is_worker():
        try:
            phys = resolve_devices(params.get("gpus"))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if len(phys) > 1:                                         # one process per GPU
            return D.launch(argv, phys)
    else:
        D.init()
    lead = D.rank() == 0                                          # rank 0 writes the files

    from . import mpf as F
    from .solver import SDP, Solver
    from . import checkpoint as CK, sdpb_output as O, state as STATE
    L = (params["precision"] + 2 + 31) // 32 + 1                # one guard limb

    if verb >= 1:
        print(f"{datetime.datetime.now():%Y-%b-%d %H:%M:%S} Start SDPB (SDPviBe {VERSION}, drop-in for SDPB {SDPB_VERSION})")
        print(f"SDPB version: {SDPB_VERSION}")
        print(f'SDP directory   : "{os.path.abspath(params["sdpDir"])}"')
        print(f'out directory   : "{os.path.abspath(out_dir)}"\n')
        print_parameters(params)
    try:
        sdp = SDP(params["sdpDir"], L)
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if verb >= 1:
        print(f"Initialize SDP solver\n\tprimal dimension: {sum(b['K'] * b['dim'] * (b['dim'] + 1) // 2 for b in sdp.layout.blocks)}"
              f"\n\tdual dimension: {sdp.N}\n\tSDP blocks: {sdp.nblocks}")
        if verb >= 2: print(f"\t{sdp.read_info}")
    parts = params["writeSolutionParts"]
    if "z" in parts and sdp.normalization is None:
        print("Error: Please provide SDP with valid normalization.json or exclude z from --writeSolution arguments.", file=sys.stderr)
        return 1
    solver = Solver(sdp, L, precision=params["precision"], **solver_params(params))
    digits = solver.digits
    try:
        st = CK.load(params["initialCheckpointDir"], sdp, L)
    except (ValueError, OSError) as e:
        print(f"Error: cannot load checkpoint from {params['initialCheckpointDir']}: {e}", file=sys.stderr)
        return 1
    if st is None and params["requireInitialCheckpoint"]:
        print(f"Error: Unable to load checkpoint from directory: {params['initialCheckpointDir']}", file=sys.stderr)
        return 1
    if st is not None:
        if verb >= 1: print(f"Loading checkpoint from : {st['source']}")
        STATE.set_state(solver, st)

    sigterm = {"got": False}
    signal.signal(signal.SIGTERM, lambda *_: sigterm.update(got=True))
    ck_opts = options_record(params)

    def save_ck(s):
        if not params["checkpointDir"]: return
        if verb >= 1: print(f"Saving checkpoint to    : {params['checkpointDir']}")
        st = STATE.get_state(s)                                     # collective
        if lead: CK.save(params["checkpointDir"], st, sdp, L, params["precision"], ck_opts)

    ij = O.IterationsJson(out_dir) if lead else None
    if verb >= 1:
        print(f"{datetime.datetime.now():%Y-%b-%d %H:%M:%S} Start solver iterations")
        print("\n          time    mu     P-obj       D-obj      gap         P-err       p-err       D-err      P-step   D-step   beta\n"
              + "-" * 117)

    def on_it(row):
        if lead: ij.append(row)
        if verb >= 1: print(iteration_line(row, solver.last_mu, solver.last_steps), flush=True)

    res = solver.run(log=None, verbose=False, on_iteration=on_it, max_runtime=params["maxRuntime"],
                     checkpoint=save_ck if params["checkpointDir"] else None,
                     checkpoint_interval=params["checkpointInterval"], stop=lambda: sigterm["got"])
    if lead: ij.close()
    reason = res["terminateReason"]
    if verb >= 1:
        f = lambda v: O.fmt_big(v, digits)
        print(f"-----{reason}-----\n\nprimalObjective = {f(res['primalObjective'])}\ndualObjective   = {f(res['dualObjective'])}"
              f"\ndualityGap      = {f(res['dualityGap'])}\nprimalError     = {f(res['primalError'])}"
              f"\ndualError       = {f(res['dualError'])}\n")
    if reason == "SIGTERM signal received" or not params["noFinalCheckpoint"]:
        save_ck(solver)
    if verb >= 1: print(f"Saving solution to      : \"{os.path.abspath(out_dir)}\"")
    state = STATE.get_state(solver) if parts else None             # collective
    cmb = STATE.c_minus_By_blocks(solver)                           # collective
    if lead:
        os.makedirs(out_dir, exist_ok=True)
        O.write_out_txt(out_dir, res, digits, time.time() - start)
        if parts:
            O.write_solution(out_dir, state, parts, digits, sdp.normalization)
        O.write_c_minus_By(os.path.join(out_dir, "c_minus_By", "c_minus_By.json"), cmb, digits)
    if reason == "SIGTERM signal received":
        if verb >= 1: print("Received SIGTERM, exiting gracefully...")
        return 15
    return 0
