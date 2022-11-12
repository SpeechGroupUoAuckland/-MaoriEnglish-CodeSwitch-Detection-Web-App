@REM start streamlit run website.py
@REM python webapi.py

conda activate dev
start python wwebapi.py
cd frontend
npm run build
npm run start
