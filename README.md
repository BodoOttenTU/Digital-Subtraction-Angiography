# Digital-Subtraction-Angiography: AIF variability in perfDSA

BEP project (TU/e, 2026) on how the arterial input function (AIF) varies in automatic DSA perfusion imaging
with [perfDSA](https://github.com/RuishengSu/perfDSA), on the PRAISE biplane DSA data.

All code is in [`perfDSA-master/`](perfDSA-master/README.md); its README has the installation, the data
layout and the full step list. In short:

```bash
cd perfDSA-master
pip install -r requirements.txt              # Python 3.14
python run_all.py --from raw_model           # full run from the DICOMs (GPU, hours)
python run_all.py                            # from post-processing on, re-using saved perfDSA runs
```

Patient data (`PRAISE/`, `PRAISE_BATCH_2/`), model weights and results are not in this repository.
