PYTHON3ML="/vol/bitbucket/395_anaconda/miniconda3/bin/python3"
echo "Setting python path to "${PYTHON3ML}
echo "Creating virtual environment under ./env"
${PYTHON3ML} -m venv ./env # Create a virtual environment (python3)
echo "Install requirements"
source env/bin/activate
which python
pip install -r requirements.txt --no-cache-dir
deactivate
unset PYTHON3ML
echo "Finished und unset python path"