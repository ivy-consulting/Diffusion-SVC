from fastapi import FastAPI, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import json
import librosa
import soundfile as sf
from tools.infer_tools import DiffusionSVC
import torch
from ast import literal_eval

app = FastAPI()

# Load configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)


@app.post("/process-audio/")
async def process_audio(
    input_wav: UploadFile,
    model_name: str = Form(...),
    output_wav: str = Form(...),
    combine_model: str = Form(None),
    keychange: int = Form(None),
    speaker_id: int = Form(None),
    speedup: int = Form(None),
    method: str = Form(None),
    kstep: int = Form(None)
):
    # Retrieve model configuration from config.json
    if model_name not in config['models']:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found in configuration.")
    
    model_config = config['models'][model_name]

    
    # Use provided values or fall back to config defaults
    combine_model = combine_model or model_config['model_path']
    keychange = keychange if keychange is not None else model_config['keychange']
    speaker_id = speaker_id if speaker_id is not None else model_config['spk_id']
    speedup = speedup if speedup is not None else model_config['speedup']
    method = method or model_config['method']
    kstep = kstep if kstep is not None else model_config['kstep']
    
    # Save uploaded file temporarily
    temp_input_path = f"/tmp/{input_wav.filename}"
    try:
        with open(temp_input_path, "wb") as f:
            f.write(input_wav.file.read())
        print("input file saved", temp_input_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {str(e)}")
    
    # Output file path
    temp_output_path = Path(output_wav)
    
    # Ensure the directory exists
    try:
        temp_output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create output directory: {str(e)}")
    
    # Load and process the audio using DiffusionSVC
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("device detected is ->", device)

        diffusion_svc = DiffusionSVC(device=device)

        diffusion_svc.load_model(model_path=combine_model, f0_model=model_config.get('pitch_extractor', "None"), f0_max=model_config.get('f0_max', "None"), f0_min=model_config.get('f0_min', "None"))
        
        print("Model loaded successfully!")
        spk_mix_dict = literal_eval(model_config.get("spk_mix_dict", "None"))
        spk_emb = None
        print("litral eval success")

        # load wav
        in_wav, in_sr = librosa.load(temp_input_path, sr=None)
        if len(in_wav.shape) > 1:
            in_wav = librosa.to_mono(in_wav)

        print("---- to enter to infer ----")
        
        # infer
        out_wav, out_sr = diffusion_svc.infer_from_long_audio(
            in_wav, sr=in_sr,
            key=float(keychange),
            spk_id=int(speaker_id),
            spk_mix_dict=spk_mix_dict,
            aug_shift=int(model_config.get('formant_shift_key', 0)),
            infer_speedup=int(speedup),
            method=config.get('method', 'dpm-solver'),
            k_step=config.get('kstep', 200),
            use_tqdm=True,
            spk_emb=spk_emb,
            threhold=float(model_config.get('threhold', 0)),
            threhold_for_split=float(model_config.get('threhold_for_split', 0)),
            min_len=int(model_config.get('min_len', 0)),
            index_ratio=float(model_config.get('index_ratio', 0))
        )
        
        # save
        sf.write(temp_output_path, out_wav, out_sr)
        return FileResponse(path=temp_output_path, filename=output_wav, media_type='audio/wav')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up the temporary input file
        try:
            os.remove(temp_input_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to clean up temporary files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
