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


# Paths to model and config
CONFIG_PATH = "high_range_models/config.json"
MODEL_PATH = "high_range_models/G_riri_220.pth"
import subprocess
from fastapi import File

@app.post("/infer/")
async def infer_audio(file: UploadFile = File(...), model_name: str = Form(...), output_wav: str = Form(...)):
    if model_name.lower() != "saotome":
        return {"error": f"Model for the actor name {model_name} is not available."}
    
    # Save uploaded file to a temporary path
    temp_input_path = f"/tmp/{file.filename}"
    temp_output_path = f"/tmp/{output_wav}"

    try:
        with open(temp_input_path, "wb") as f:
            f.write(file.file.read())
        print("input file saved", temp_input_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {str(e)}")

    # Run inference
    try:
        command = [
            "svc", "infer", temp_input_path, 
            "-c", CONFIG_PATH, 
            "-m", MODEL_PATH
        ]
        subprocess.run(command, check=True)



        return FileResponse(temp_output_path, media_type="audio/wav", filename=output_wav)
    except Exception as e:
        return {"error": f"Inference failed, {e}"}
    finally:
        # Clean up temporary files
        try:
            os.remove(temp_input_path)
            print("input file removed", temp_input_path)
        except Exception as e:
            return {"error": f"Failed to clean up temporary files, {e}"}

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
    kstep: int = Form(None),
    formant_shift_key: int = Form(0),
    pitch_extractor: str = Form("fcpe"),
    f0_min: int = Form(50),
    f0_max: int = Form(1100),
    threhold: int = Form(-60),
    threhold_for_split: int = Form(-40),
    min_len: int = Form(5000),
    index_ratio: int = Form(0)
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
    formant_shift_key = formant_shift_key if formant_shift_key is not None else model_config['formant_shift_key']
    pitch_extractor = pitch_extractor or model_config['pitch_extractor']
    f0_min = f0_min if f0_min is not None else model_config['f0_min']
    f0_max = f0_max if f0_max is not None else model_config['f0_max']
    threhold = threhold if threhold is not None else model_config['threhold']
    threhold_for_split = threhold_for_split if threhold_for_split is not None else model_config['threhold_for_split']
    min_len = min_len if min_len is not None else model_config['min_len']
    index_ratio = index_ratio if index_ratio is not None else model_config['index_ratio']
    
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

        diffusion_svc.load_model(model_path=combine_model, f0_model=pitch_extractor, f0_max=f0_max, f0_min=f0_min)
        
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
            aug_shift=int(formant_shift_key),
            infer_speedup=int(speedup),
            method=method,
            k_step=int(kstep),
            use_tqdm=True,
            spk_emb=spk_emb,
            threhold=int(threhold),
            threhold_for_split=int(threhold_for_split),
            min_len=int(min_len),
            index_ratio=int(index_ratio)
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
