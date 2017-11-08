# Bioinformatics

This is a project for Bioinformatics course.
Course webpage is located on the following link.
http://www.fer.unizg.hr/en/course/bio

Project subject is calculating local alignment using Smith-Waterman algorithm implemented on CUDA platform.


Team members: Dario Sitnik, Franjo Matković. Matej Crnac

# Installation of project dependencies

Follow the next instructions for Ubuntu 16.04:

Install CUDA driver (version 384) on Ubuntu 16.04:

1) if there are older drivers, remove them:
	$ sudo apt-get purge nvidia* 

2) add graphics drivers PPA:
	$ sudo add-apt-repository ppa:graphics-drivers

3) update it:
	$ sudo apt-get update

4) install latest driver:
	- to find the latest available driver use the following command:
	$ cat /var/lib/apt/lists/ppa.launchpad.net_graphics-drivers_*_Packages | grep "Package:" | sort | uniq
	
	$ sudo apt-get install nvidia-387

5) reboot the computer and check if the driver has installed correctly:
	$ reboot
    
	$ lsmod | grep nvidia #if this shows something then driver has installed

	$ lsmod | grep nouveau #if this shows something and the above shows nothing then it has not installed correctly

6) use command to stop system for automatically updating:
	$ sudo apt-mark hold nvidia-387

7) if necessary, you can remove driver with the next command:
	$ sudo apt-get purge nvidia*
	$ reboot #to install open-source nouveau drivers

Installing CUDA toolkit (version 8) on Ubuntu 16.04:

Go to the following link:
https://developer.nvidia.com/cuda-80-ga2-download-archive

and select next properties:
	Operating System: Linux
	Architecture: x86_64
	Distribution: Ubuntu
	Version: 16.04
	Installer Type: deb(local)

After downloading baseinstaller follow next installation instructions:
	1) $ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
	2) $ sudo apt-get update
	3) $ sudo apt-get install cuda

Post installation steps:
	1) $ sudo nano /etc/environment
	Add "/usr/local/cuda-8.0/bin" to PATH
	2) $ source /etc/environment  #so it is immediately updated
	To see if it installed correctly:
	3) $ nvcc --version


