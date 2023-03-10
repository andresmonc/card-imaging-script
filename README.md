# card-imaging-script

This script detects playing cards against a black background and splits them out into multiple organized front/back(A/B)
image files.

# How to Use

- Place the 2 scans (front/back) in input
- Run the script
- Enjoy your split photos (saved in output)


# How to Build
- install pyinstaller
- run command below in cardimaging directory
pyinstaller --onefile --paths=C:\Users\Andresmonc\PycharmProjects\card-imaging-script\venv\Lib\site-packages .\main.py
