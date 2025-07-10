# Project Overview
This is a multimodal AI application built with python to analysize text, audio and video. It allows users to add , delete, and mark tasks as complete.


# Code Style
- Use 2-space indentation.
- Use modules when appropriate for easyly maintaining code.

# Tools and Libraries
- openai
- webrtcvad
- gradio


# Current Goals
- use graio to create UI to input video, adjuest parameters and ouput results
- The input video can be a mp4 file or video recorded from camera, the video cvan be replay with UI
- the adjuestable parameters including  video path and initial prompt to described by user
- the output should inlcude SEGEMENT output and final response ouput. 
- Refactor `v2.3-segements-hazard-region.py` for better User Experience using gradio.

# Key Files
- `v2.3-segements-hazard-yolo.py`: Main python file.
- `proposal.md`

# Agent Instructions
- Do not modify other file except `v2.2-segements-openai-ui.py`.
- Prioritize user interface.
- Use comments to explain complex logic.
- Use "uv run" to run python code.
