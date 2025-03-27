# ZoneIn

## Overview
ZoneIn is an intelligent eye-tracking and concentration monitoring system that leverages computer vision and machine learning to analyze user focus and attention. Using facial landmark detection and eye aspect ratio analysis, the system provides real-time insights into a user's concentration levels.

## Problem Statement
Traditional focus tracking methods are limited by:
* Inability to accurately measure sustained attention
* Lack of real-time feedback on concentration
* Minimal insights into distraction patterns
* Limited capabilities in understanding user engagement

## Key Features
* **Intelligent Eye Tracking**
   * Advanced facial landmark detection
   * Precise eye aspect ratio (EAR) calculation
   * Adaptive baseline calibration
* **Comprehensive Focus Monitoring**
   * Real-time focus status detection
   * Detailed distraction logging
   * Customizable tracking parameters
* **Data Persistence**
   * MongoDB integration for session tracking
   * Comprehensive focus reports
   * Historical performance analysis

## Technology Stack
Computer Vision
* OpenCV
* dlib
* NumPy

Machine Learning
* Facial Landmark Detection
* Eye Aspect Ratio (EAR) Algorithm

Database
* MongoDB

Additional Technologies
* Scipy
* Matplotlib
* Threading

## System Requirements
* Python 3.7+
* MongoDB
* dlib
* OpenCV
* CUDA-compatible GPU (recommended)

## Installation

1. Clone the Repository
```bash
git clone https://github.com/SSARAWAGI05/ZoneIn.git
cd ZoneIn
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Download Facial Landmark Predictor
* Download `shape_predictor_68_face_landmarks.dat`
* Place in project root directory

4. Configure MongoDB
* Ensure MongoDB is installed and running
* Update connection details in the script if needed

5. Run the System
```bash
python app.py
```

## Configuration
The system automatically calibrates eye tracking, but you can customize parameters in the code:
* Adjust calibration frame count
* Modify baseline EAR thresholds
* Configure MongoDB connection settings

## Usage
* Run the script
* Position yourself in front of the camera
* The system will track your focus in real-time
* Press 'q' to exit and generate a focus report

## Ethical Considerations
* Respects user privacy
* Non-invasive tracking
* Designed for personal productivity enhancement

## Potential Applications
* Productivity Monitoring
* Learning Assistance
* Remote Work Engagement
* Academic Performance Tracking
* Cognitive Research

## Future Roadmap
* Enhanced machine learning models
* Cloud synchronization
* More detailed focus analytics
* Cross-platform support
* Integration with productivity tools

## Contributing
Contributions are welcome! 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Limitations
* Requires direct camera view
* Performance depends on lighting conditions
* Calibration might vary between users

## Disclaimer
This tool is intended for personal productivity and research. Always respect individual privacy and obtain necessary consents.

## License
MIT License

## Contact
Shubam Sarawagi
sarawagishubam@gmail.com
Project Link: https://github.com/SSARAWAGI05/ZoneIn

## Acknowledgements
* dlib Community
* OpenCV Contributors
* MongoDB Developers
