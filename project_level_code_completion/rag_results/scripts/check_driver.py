import os
import tempfile

from hydra import compose, initialize_config_dir

from eval import compare_arms

CONF = os.path.join(os.environ["PROJECT"], "eval", "config")
with initialize_config_dir(config_dir=CONF, version_base=None):
    cfg = compose(config_name="compare_arms", overrides=["arm=rag", "limit_datapoints=20"])
    print("arm:", cfg.arm, "| model:", cfg.params.inference_params.model, "| seq_max_len:",
          cfg.params.inference_params.seq_max_len, "| dataset:", compare_arms.dataset_name(cfg))
    print("arm dir:", compare_arms.arm_directory(cfg))
    print("separators:", repr(cfg.composers_config.extension), repr(cfg.composers_config.lang_sep_symbol),
          repr(cfg.composers_config.meta_info_sep_symbol))
    print("budget:", cfg.params.inference_params.seq_max_len - compare_arms.GENERATION_RESERVE)

    bad = compose(config_name="compare_arms", overrides=["arm=none"])
    try:
        compare_arms.validate(bad)
    except ValueError as e:
        print("bad arm rejected:", str(e)[:70])

    ok = compose(config_name="compare_arms", overrides=["arm=path_distance", "arm_name=pd_test"])
    compare_arms.validate(ok)
    print("static arm dir:", compare_arms.arm_directory(ok))

    # Existing results must not be appended to unless overwrite=true.
    with tempfile.TemporaryDirectory() as tmp:
        guarded = compose(config_name="compare_arms", overrides=["arm=rag", f"artifacts_dir={tmp}"])
        d = compare_arms.arm_directory(guarded)
        d.mkdir(parents=True)
        (d / "generation_results.jsonl").write_text("{}\n")
        try:
            compare_arms.run_arm(guarded)
        except FileExistsError as e:
            print("existing results protected:", str(e)[-45:])
        print("results file untouched:", (d / "generation_results.jsonl").read_text() == "{}\n")
