# Bioinformatics

This is a project for Bioinformatics course.
Course webpage is located on the following link.
http://www.fer.unizg.hr/en/course/bio

Project subject is calculating local alignment using Smith-Waterman algorithm implemented on CUDA platform.


Team members: Dario Sitnik, Franjo Matković. Matej Crnac

#Installation of project dependencies

We did it following the next instructions:

Installing cuda driver (version 384) on Ubuntu 16.04:

1) if there were older drivers, we removed them:
	$ sudo apt-get purge nvidia* 

2) then we added graphics drivers PPA:
	$ sudo add-apt-repository ppa:graphics-drivers

3) updated it:
	$ sudo apt-get update

4) installed latest driver:
	$ sudo apt-get install nvidia-387

-to find the latest available driver we used the following command:
cat /var/lib/apt/lists/ppa.launchpad.net_graphics-drivers_*_Packages | grep "Package:" | sort | uniq

5) rebooted computer and checked if the driver has installed correctly:
	$ reboot
	
	$ lsmod | grep nvidia #if this shows something then deriver has installed

	$ lsmod | grep nouveau #if this shows something and the above shows nothing than is has not installed correctly

6) we used command to stop system for automatically updating:
	$ sudo apt-mark hold nvidia-387

7) if necessary, you can remove driver with the next command:
	$ sudo apt-get purge nvidia*
	$ reboot #to install open-source nouveau drivers

Installing cuda toolkit (version 8) on Ubuntu 16.04:

We went to the following link:
https://developer.nvidia.com/cuda-80-ga2-download-archive

and selected next properties:
Operating System: Linux
Architecture: x86_64
Distribution: Ubuntu
Version: 16.04
Installer Type: deb(local)

After downloading baseinstaller we followed next installation instructions on the webpage:
1) $ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
2) $ sudo apt-get update
3) $ sudo apt-get install cuda

Post installation steps:
1) $ sudo nano /etc/environment
We added "/usr/local/cuda-8.0/bin" to PATH
2) $ source /etc/environment  #so it is immediately updated
To see if it installed correctly:
3) $ nvcc --version


