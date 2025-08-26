# 0) If your venv is active, deactivate it first
deactivate 2>/dev/null || true

# 1) Clean module env and load Arrow *before* activating your venv
module purge
module load StdEnv/2023 gcc            # adjust if your site uses a different StdEnv
module avail arrow                     # see versions             # <-- pick the newest shown by 'module avail arrow'
module load python/3.10                # same Python you used for the venv
module load gcc arrow/16.1.0
# 2) Activate your env
source /scratch/yujingz/HECKTOR25/envs/hecktor25/bin/activate

# 3) Remove any broken pyarrow bits that may have been left in your env
# pip uninstall -y pyarrow pyarrow-noinstall pyarrow_noinstall || true

# 4) Install Ray (choose what you actually need)
# If you need Ray Data (requires PyArrow):
pip install -U "ray[all]"

# 5) Sanity checks
python - << 'PY'
import pyarrow, ray
print("pyarrow:", pyarrow.__version__, "->", pyarrow.__file__)
print("ray:", ray.__version__)
PY

# optional: (or can install only the essential ones manually)
pip install -r /scratch/yujingz/HECKTOR25/GaMMA-Surv/requirements.txt