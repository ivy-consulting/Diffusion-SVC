from fastapi import FastAPI, Form, UploadFile, HTTPException
import subprocess
from pathlib import Path
import os

app = FastAPI()

@app.post("/process-audio/")
async def process_audio(
    input_wav: UploadFile,
    output_wav: str = Form(...),
    combine_model: str = Form(...),
    keychange: int = Form(0),
    speaker_id: int = Form(1),
    speedup: int = Form(10),
    method: str = Form("dpm-solver"),
    kstep: int = Form(200)
):
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
