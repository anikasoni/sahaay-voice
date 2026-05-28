# Stable Windows run instructions

This build pins Streamlit to 1.40.2 because the newer 1.57 audio widget/server path was observed to exit after one browser mic recording on Windows without a Python traceback.

Run from PowerShell:

```powershell
C:\Users\Anika\Downloads\sahaay_voice\sv\Scripts\Activate.ps1
cd C:\Users\Anika\Downloads\sahaay_voice_phase1_8_streamlit140_stable
python -m pip uninstall -y streamlit
python -m pip install streamlit==1.40.2
(Get-Content requirements.txt) -replace '^webrtcvad>=2.0.10','webrtcvad-wheels>=2.0.10' | Set-Content requirements.win.txt
python -m pip install -r requirements.win.txt
python -m streamlit run app/main.py --server.fileWatcherType none --server.runOnSave false
```

Do not use Streamlit 1.57 for browser-mic calibration.
