from fastapi import FastAPI, Form, UploadFile, HTTPException
import subprocess
from pathlib import Path
import os
import json

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {str(e)}")
    
    # Output file path
    temp_output_path = Path(output_wav)
    
    # Ensure the directory exists
    try:
        temp_output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create output directory: {str(e)}")
    
    # Command to execute
    command = [
        "python", "main.py", 
        "--model_name", model_name,
        "-i", temp_input_path,
        "-model", combine_model,
        "-o", str(temp_output_path),
        "-k", str(keychange),
        "-id", str(speaker_id),
        "-speedup", str(speedup),
        "-method", method,
        "-kstep", str(kstep)
    ]
        
    # Execute the command
    try:
        subprocess.run(command, check=True)
        return {"message": "Processing complete", "output_file": str(temp_output_path)}
    except subprocess.CalledProcessError as e:
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
