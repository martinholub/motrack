# motion-tracking
Repository for code for tracking the motion of mice during behavioral tests.

----

# Installation

1. [Download](https://conda.io/miniconda.html) and install Miniconda with Python 3.6  _(Skip if you already have Miniconda installed.)_
2. Open Anaconda Prompt and type: `conda env create -f motion-tracking\py36.yml` where the last argument is relative path to the environment configuration file with respect to your current working directory. This will install a new Python environment with necessary setup.
3. In Anaconda Prompt type `activate py36` to activate your newly installed environment

# Usage

#### Overall pipeline

``` shell
cd /path/to/motion-tracking
python process.py
python analyze.py
```

#### Detailed pipeline

1. Convert videos to `.avi` file format with `800x600` resolution using for example [Free MP4 Video Converter](https://www.dvdvideosoft.com/products/dvd/Free-MP4-Video-Converter.htm) or [ffmpeg](https://www.ffmpeg.org/)
2. Put videos to be analyzed into `motion-tracking\vids` folder, add also `start_stop.csv` file. You may need to check code in `process.py` file to make sure that your file looks as expected (`sep = ";"`, `skip_rows = 2`, `.MP4` extension after filename).
3. Open *Anaconda Command Prompt*, locate to `motion_tracking` directory and type `python process.py`
  * First video that is processed will prompt you to pick ROI and range of pixel values in hsv_space
    * You should select a ROI much bigger (~5x) than the animal, and you may make it bigger in the direction of anticipated movement
    * For HSV range you may try (`H: 0-179`, `S: 0-50`, `V: 160-255`)
  * All other videos in the `vids` folder are processed using the information you provided in `start_stop.csv` and during the initial user interaction
    * You can inspect the tracking visually or you can turn off the plotting by adjusting `show_frames` and `show_mask` parameters in files `params_init` (applies to 1st video), or `params.py` (applies to all other videos)

# Results

Results of the tracking are stored in `res` folder. You will find the total time and distance together with eventual warnings at the bottom each `.txt` file. To collect aggregate information for all the files corresponding to all your videos in `vids` folder, you type `python analyze.py` in the command prompt. The result will be a single file in your `motion-tracking` directory with the aforementioned results pulled out.

### Intermediate parameters

Several parameters get saved during the processing of each video. You will find them in `inits` folder. They should enable you to reproduce the whole analysis without the initial user interaction (i.e. the results should be the same for all such runs). To do that call `python process.py --reproduce 1`

# Tests 

You may test that the code runs as expected by invoking `python -m pytest`. You will need to put necessary files to `tests\res` folder and you can find them in `~\EIN Group\MartinHolub\motion-tracking\tests\res`.

### Reporting issues

If you spot a bug or something is not working as you would expect, you can go to [issues page](https://github.com/EIN-lab/motion-tracking/issues) and create an issue where you descibe what is the problem and what lead to its occurence.

