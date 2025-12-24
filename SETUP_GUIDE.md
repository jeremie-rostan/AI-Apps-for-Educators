# Step-by-Step Setup Guide for Educators

This guide will help you set up and run the Teacher Evaluation AI system on your own computer, even if you have no technical background.

## What You'll Need

1. A computer (Windows, Mac, or Linux)
2. About 15-20 minutes
3. A Google API key (free - instructions below)

## Step 1: Install Python

Python is the programming language that runs this application.

### For Mac:
1. Open your web browser and go to [python.org/downloads](https://www.python.org/downloads/)
2. Click the yellow "Download Python" button
3. Open the downloaded file and follow the installation steps
4. Keep clicking "Continue" and "Install"

### For Windows:
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Click the yellow "Download Python" button
3. **Important:** When the installer opens, check the box that says "Add Python to PATH"
4. Click "Install Now"

### Verify Python is installed:
1. Open Terminal (Mac) or Command Prompt (Windows)
   - **Mac:** Press `Cmd + Space`, type "Terminal", press Enter
   - **Windows:** Press `Windows key`, type "cmd", press Enter
2. Type: `python --version` and press Enter
3. You should see something like "Python 3.12.0"

## Step 2: Get a Google API Key

You need a free API key to use Google's AI models.

1. Go to [aistudio.google.com](https://aistudio.google.com/)
2. Click "Sign in" and use your Google account
3. Click "Get API key" in the top right
4. Click "Create API key"
5. Click "Create API key in new project"
6. **Important:** Copy this key and save it somewhere safe (you'll need it in Step 5)

## Step 3: Download the Project

1. Go to [github.com/jeremie-rostan/agno-agents-for-educators](https://github.com/jeremie-rostan/agno-agents-for-educators)
2. Click the green "Code" button
3. Click "Download ZIP"
4. Find the downloaded ZIP file (usually in your Downloads folder)
5. Double-click to unzip it
6. Move the unzipped folder to somewhere easy to find (like your Desktop or Documents)

## Step 4: Install Required Software

1. Open Terminal (Mac) or Command Prompt (Windows) again
2. Navigate to the project folder:
   - **Mac:** Type `cd ` (with a space), then drag the folder from Finder into the Terminal window, then press Enter
   - **Windows:** Type `cd ` (with a space), then drag the folder from File Explorer into the Command Prompt window, then press Enter
3. Install the required packages by typing this command and pressing Enter:
   ```
   pip install -r requirements.txt
   ```
4. Wait for it to finish (this might take a few minutes)

## Step 5: Add Your Google API Key

1. In the project folder, find the file called `.env.example`
2. Make a copy of this file and rename it to `.env` (just remove the `.example` part)
   - **Mac:** Right-click the file, choose "Duplicate", then rename
   - **Windows:** Right-click the file, choose "Copy", then "Paste", then rename
3. Open the `.env` file with a text editor:
   - **Mac:** Right-click → Open With → TextEdit
   - **Windows:** Right-click → Open With → Notepad
4. Replace `your_api_key_here` with the Google API key you copied in Step 2
5. Save and close the file

## Step 6: Run the Application

1. In Terminal or Command Prompt, make sure you're still in the project folder
2. Navigate to the teacher evaluation example:
   ```
   cd examples/teacher-evaluation
   ```
3. Start the application:
   ```
   python app.py
   ```
4. You should see text saying the server is running at `http://localhost:7777`

## Step 7: Use the Application

1. Open your web browser
2. Go to: `http://localhost:7777`
3. Optional: connect to the [AgentOS interface](https://os.agno.com/) 
4. You can now submit teacher evaluations and receive professional development reports!

## Troubleshooting

### "Python is not recognized" or "command not found"
- You need to install Python (go back to Step 1)
- On Windows, make sure you checked "Add Python to PATH" during installation

### "pip is not recognized" or "command not found"
- Try using `python -m pip install -r requirements.txt` instead

### "No module named 'agno'"
- Make sure you ran the `pip install -r requirements.txt` command in Step 4

### The browser shows "This site can't be reached"
- Make sure the application is still running in Terminal/Command Prompt
- Try the address `http://127.0.0.1:7777` instead

### API key errors
- Double-check that you copied the entire API key correctly
- Make sure your `.env` file is in the main project folder, not still named `.env.example`

## Stopping the Application

When you're done:
1. Go back to the Terminal or Command Prompt window
2. Press `Ctrl + C` (on both Mac and Windows)
3. The application will stop

## Getting Help

If you encounter issues:
1. Check the [Agno Documentation](https://docs.agno.sh)
2. Create an issue on [GitHub](https://github.com/jeremie-rostan/agno-agents-for-educators/issues)
3. Make sure all steps were followed exactly as written

## What's Next?

Once you have it running, you can:
- Submit teacher evaluations through the web interface
- Receive detailed professional development reports
- Customize the ISP_Way.txt file to match your institution's teaching standards
