# SightCom-2

For more information, visit: [SightCom 2 Article](https://lablab.ai/event/ai-challenge-with-gpt-3-5-codex-dall-e-and-whisper-api/louis/sightcom-2)

## Overview
Around the world, there exists an inequality affecting visually impaired individuals who lack access to essential accessibility services. This has driven me to develop a software for Smart Glasses that blind people can wear.
SightCom2 software utilizes OpenAI technologies, namely Whisper for speech transcription, GPT-3.5 as a LLM, DALL-E for image generation; image captioning, OCR and color recognition models from Clarifai API. 
This software is served on streamlit cloud, and is a prototype that can potentially be deployed on a microprocessor, assembled in an integrated circuit, between input devices like camera and microphone, and output devices like speakers. 

## How it works? 
![SightCom 2 Flowchart](https://github.com/Louisljz/SightCom-2/blob/main/SightCom%202%20FlowChart.png)

## Run App Locally
1. **Clone the Repository**:  
   ```
   git clone https://github.com/Louisljz/SightCom-2.git
   ```

2. **Install Virtual Environment**:  
   ```
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows: `venv\Scripts\Activate`
   - On macOS and Linux: `source venv/bin/activate`

3. **Install Required Packages**:  
   ```
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:  
   ```
   streamlit run app.py
   ```

5. **Open the App**:  
   The app should now be running. Open your web browser and go to `http://localhost:8501/` to interact with the app.
