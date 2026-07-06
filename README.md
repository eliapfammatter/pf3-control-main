# Setup

## General

### Python setup

Use python 3.13.

### Data

The data, which is in some cases sensitive, is _not_ in the git repository, it can be downloaded [here](https://drive.switch.ch/index.php/s/bRH4qxGvv7tWvt6).
This directory should then be copied into the project root.

## SIMSEN FMU setup
SIMSEN only runns on WINDOWS. However, you can run it as remote FMU on windows and run a thin wrapper on linux side to acces the windows server.
This means, if you want to use FMU on MacOS or Linux, you need to have access to a windows machine running the SIMSEN RFMI server.
The way how this is done does not really matter, but the machine executing the FMU script, should reach the application under the IP and PORT specified in the ENV variable `RFMI_SERVER_SIMSEN="192.168.122.47 6090"`.
As a further caviate, this remote FMU _only_ works on Linus, not MacOS. As a workaround, this repository offers a wrapper, `run-on-linux.sh`, which can run any command in a Linux Docker. If you thus want to use FMI, you can use this wrapper. The wrapper however has caviates, it does not have a display, and thus plotting does not work. It should just be used to generate the FMI output, and everything else should be run natively.

How to setup this system is described below.

### Windows access
SIMSEN, which is used for FMI, only runns on Windows. If on MacOS or Linux, install a Windows VM, or make sure you can access a remote machine with SIMSEN installed.

#### Install a VM on MacOS
Use a virtualizer, for example the open-source solution UTM: https://mac.getutm.app/

During installation, install CrystalFetch ISO Downloader from mac app store for the ISO download of windows.

Download latest Windows 11.

During install, press any key instantly as soon as you see the message. If you land in the `Shell` mode, stop the VM and start again. After the first install wizard, do not press any key anymore so you start into the actual VM and not into the install medium.

### Install SIMSEN

Install SIMSEN according to the SIMSEN installation guide.

Install it into `C:\Users\<USERNAME>\SIMSEN`.

### Setup FMI

In general, refering to help doc FMI.pdf.
All SIMSEN software lives in `C:\Users\<USERNAME>\SIMSEN\SIMSEN_4_0_3_2025_05\exe`

- [ ] run `InstallSimsen4FMI.bat`, which can be found in the SIMSEN software path. (as described on page 16)
- [ ] run the `start-rfmi-wrapper` script (from `scripts`) on windows side, allow network access. This is an automatic restart script, which always restarts the `SimsenRFMIServer`, if it is hanging. The `SimsenRMFIServer` will start a server which listens on port 6090.
- [ ] in comand prompt (`cmd.exe`) run `ipconfig` to check ip of machine. (in this example it will be `192.168.122.47`)
- [ ] if needed, in command prompt, run `netstat -an | findstr 6090` to check if SIMSEN RFMI SERVER is listening.
- [ ] go to `start -> Windows Security -> Firewall & network protection -> Allow an app through firewall` and set both checkmarks, on `private` and `public` for `simsenrfmiserver.exe`.
- [ ] then check if you can connect to the windows machien from your master machine with e.g. `nc -zv 192.168.122.47 6090`.
- [ ] Add RFMI_SERVER_SIMSEN to your environment variables, e.g. probably in your bash.rc (or zsh.rc for mac and specific linux), add in the last line `export RFMI_SERVER_SIMSEN="192.168.122.47 6090"
- [ ] When running FMU, make sure that the server is restarted cleanly bevor each run. This should be handeled by the restart scirpt. And wait a few seconds after restart of the server and starting the co-simulation.
- [ ] If you are running on MacOS, you need to run the FMU scripts from a linux Docker. To do so, use the `run-on-linux.sh` wrapper, e.g. `./run-on-linux.sh python compare_simulation_models.py --stop-time 60 --skip-native --skip-casadi --skip-scipy --no-plot`. Having Docker installed on your Mac is a prerquisite. This wrapper is only needed if you run FMU. e.g. if you have cached the FMU output already, or you are skipping it, you can also run it natively on MacOS. (e.g. for plotting).

# Housekeeping

## Package management

Package management is done with pip, version pinning platform-specific with `pip-compile`.

If you want to install a new package, add it to `requirements.in` (or `requirements-dev.in` if it is only a dev-tool). Then run `pip-compile requirements.in -o requirements-<PLATFORM>.txt` to update the platform-specific locked requirements file.


# Overview

1. `compare_simulation_models.py` - Compares different simulation models (140-state native, 2-state CasADi ROM, 2-state SciPy ROM) against the SIMSEN FMU reference, plotting turbine head/flow and computing error metrics.
2. `run_fmi_control.py` - Runs closed-loop FMI simulations with various controllers (FPOINTS open-loop, PI, NMPC) against the SIMSEN FMU, comparing controller performance and exporting pump speed trajectories to FPOINTS DAT files.
3. `generate_sysid.py` - Generates experiment setups by computing operating point sequences (e.g., BEP→part-load transitions at specific swirl numbers), running NMPC simulations, and exporting control trajectories (N_T, y_T, N_P) for use on the physical test rig.
4. `pf3_post_processing.py`- Post-processes experimental TDMS data by aligning measurements to simulation results, cropping to common time windows, and creating comparison plots including time-series, hillcharts, and PCB pressure sensor visualizations.

