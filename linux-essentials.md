# Linux essentials

**Contents**

* [Interacting with Linux](#interacting-with-linux)
    * [Navigating up and down the hierarchy](#navigating-up-and-down-the-hierarchy)
    * [Copying files and folders](#copying-files-and-folders)
    * [Copying files and folders between computers](#copying-files-and-folders-between-computers)
    * [(Re)moving files and folders](#removing-files-and-folders)
    * [File properties](#file-properties)
    * [Disk usage](#disk-usage)
* [Running programs and scripts](#running-programs-and-scripts)
    * [Quitting](#quitting)
    * [Pausing](#pausing)
    * [When <kbd>Ctrl-C</kbd> is not enough](#when-ctrl-c-is-not-enough)
* [Running Python scripts on GPUs](#running-python-scripts-on-gpus)
    * [Running your script on a selected GPU in TensorFlow (with or without Keras)](#running-your-script-on-a-selected-gpu-in-tensorflow-with-or-without-keras)
    * [Limiting memory use in TensorFlow (with or without Keras)](#limiting-memory-use-in-tensorflow-with-or-without-keras)
* [Keep programs and scripts running after logging out](#keep-programs-and-scripts-running-after-logging-out)
* [Installing additional programs and packages](#installing-additional-programs-and-packages)
* [More on file permsissions](#more-on-file-permissions)
* [Running your own programs](#running-your-own-programs)
* [The PATH variable and ~/.bashrc file](#the-path-variable-and-bashrc-file)
* [Extras](#extras)

During one of your projects or courses you may have to work with Linux. One prominent example is when you will use a GPU-server for deep learning experiments. Linux is an operating system (OS), like Windows, Android, or iOS. There are many distributions (think 'versions') of Linux, and you will probably use a distribution called Ubuntu.

If you are using Linux on your own PC you can use the graphical user interface, which has the windows, text fields, and application windows that you are used to. However, in most cases, you will interact with Linux using a Terminal: a text-based interface on which you can type commands. When using a remote server, you will have no choice but to use the Terminal.

This chapter has no explicit exercises, as it works best as a tutorial: just follow along the instructions in the text.

We will assume you work with a Terminal application to interact with a Linux machine. If you use your own PC with Linux, you can start the Terminal application and skip the next section. If you are going to use a Linux server, you will need the steps outlined [on this page](server-accounts.md) to connect to the server first.

## Interacting with Linux

Start the terminal application of your computer i.e. Cygwin on Windows, Gnome-Terminal in Ubuntu Linux, or Terminal.app in macOS.

A black screen with white text (or a white screen with black text if you are using macOS) will appear. There is probably a blinking cursor and a `$`-sign visible on the screen. There may be some text in front of the dollar sign like the user name or computer name that you can ignore for now. Because it is different on every system, we are going to just use `$` throughout this tutorial. The `$`-sign is called the Bash 'prompt'. Bash is the program that you run the commands in. From this prompt you can type commands. When your command is finished you can press enter to see the result, which will be printed below the command you typed.
For example, if you type `whoami` and press enter, your user name will appear. 
Below that you get a new prompt on which you can enter new commands. If you type `date` it will display the current date and time.

```bash
$ whoami
amalia

$ date
Fri Feb  8 09:35:28 CET 2019
```

Although it is nice to be able to recall your name should you forget it, or see what time and day it is, there are far more useful commands in Linux. Most of these commands let you interact with files on your PC or on a remote server. Whether you use them on a server, or use them on your local Linux PC, they behave the same way.

Like any operating system, Linux works with hierarchies of file folders, also called directories in Linux. At the top level of this hierarchy, or 'root', you have a folder called `/`. From there, you have several subfolders. One of them is called `/home` in which the user folders are. If you have an account on the computer, there should be one with your user name called `/home/amalia` for example. In that 'home folder' you can save your files, your code, and if there is enough space (inquire with the administrator if you want to be sure), your data. Because you so often use your home folder, there is a shorter way of typing it: a tilde or `~`. When in Linux the promp shows this tilde, it means that you are currently in that folder. You can also check with the `pwd` command (short for print/present working directory):

```bash
$ pwd
/home/amalia
```

In the directory you can inspect files and folders, run programs, and create, move, and delete files or folders. If you want to see what the content of the current directory are, you use the `ls` command. `ls` will show you a list of the contents of the folder. There is a chance that this folder is currently empty, so it will probably show nothing. Let's change that by typing `mkdir my_first_folder`. `mkdir` stands for 'make directory'. If you type `ls` again, you will see it shows the new folder:

```bash
$ ls

$ mkdir my_first_folder

$ ls
my_first_folder

```

Wonderful. Now you can navigate to this folder by typing `cd my_first_folder`.

```bash
$ cd my_first_folder

$ 

```

In here, let's make a new text file using the `touch` command:

```bash
$ touch my_first_file.txt

$ ls
my_first_file.txt
```

This file will be empty. We can edit it using the `gedit` command, which will open the file in the Linux text editor. You can type something here, and save it using the File menu in the gedit window.

---

Note, that gedit runs as graphical application, which does not always work well. There are also alternative editors with text-based interfaces that run inside the Terminal window, such as `nano`, `pico`, and `vim`. `nano` and `pico` can be operated using the shortkeys shown at the bottom of the window (where ^ means <kbd>Ctrl</kbd>). For example, to exit the editor use <kbd>Ctrl+X</kbd> and to save the file <kbd>Ctrl+O</kbd>. 

The `vim` editor is a story in itself. It is a very powerful editor which can be operated using a small programming language. It has a very steep learning curve. Should you ever open it by accident, know that you can quit it by pressing <kbd>Esc</kbd> a few times, and typing `:q` followed by enter. Makes sense right?

---

Now your file contains some text, and you can show it in the Terminal using the `cat` or `more` commands (e.g. `cat my_first_file.txt`). The `more` command shows the file page-by-page that you can page through using the <kbd>Enter</kbd> key.


### Navigating up and down the hierarchy

We have seen that you can navigate down the hierarhcy by typing `cd` with the folder's name. However, you can also change directory using the full path:

```bash
$ cd ~

$ cd ~/my_first_folder

$
```

If you want to move one level *up* you can use `cd ..`. `..` always denotes the parent folder of the current working directory, i.e.

```bash
$ cd ..

$ mkdir my_second_folder

$ ls
my_first_folder     my_second_folder

$ cd my_first_folder

$ cd ../my_second_folder

$
```

The last command combines both concepts: "move one folder up the hierarchy and change to the one with the name 'my_second_folder'."
Lastly, if you just type `cd` you end up in your home folder: `cd` is short for `cd ~`.

---

Tip
: The <kbd>Tab</kbd>-key can be used for autocompletion of file paths. If you type `cd ~/my_` followed by <kbd>Tab</kbd> the terminal will give suggestions for completion of your command. This can save a lot of time!

---


### Copying files and folders

You can copy files using the `cp` command. Behind the command you type one or more files you want to copy, and the destination. The destination can be a new file name or a destination folder:

```bash
$ cp my_first_file.txt copy_of_file.txt

$ ls
copy_of_file.txt    my_first_file.txt

$ cp my_first_file.txt copy_of_file.txt ../my_second_folder

$ cd ../my_second_folder

$ ls
copy_of_file.txt    my_first_file.txt

```

If the destination is a folder, you can also copy multiple files (or folders) to that destination, e.g. `cp file1 file2 destination`.

If you want to copy a folder with its content, you use the `cp` command with a `-r` flag, which means `recursive` copy. Recursive copying also copies everything *inside* a folder.

```bash
$ cd

$ cp -r my_first_folder copy_of_first_folder

$ ls
copy_of_first_folder    my_first_folder     my_second_folder
```

**Be careful!** If the destination folder does not exist yet, the folder's copy will get the name of the destination you specify. However, if the destination folder *does* exist, the folder you copy is copied *into* the destination folder. Note the difference in the following example in which the destination already existed:

```bash
$ cp -r my_first_folder my_second_folder

$ cd my_second_folder

/my_second_folder$ ls
copy_of_file.txt    my_first_file.text     my_first_folder

```


### Copying files and folders between computers
If you already have code or data on you own PC, there will probably come a time when you want to copy the files from your PC to the remote server. Or maybe you want to do the reverse. 

For this you can use the `scp` command, which stands for secure copy. Instead of only specifying filenames, you specify from or to which computer as well. You *need* to run this command from a terminal window *on your own PC*, so without logging into the server. You can get a new terminal window from the terminal application's menu. Check if the new terminal window is indeed displaying your own machine.

Then, you navigate to the folder you want to copy from or to *on your PC*. Next, you type either

```bash
$ scp -rp /folder/on/your/pc  amalia@saruman.bmt.tue.nl:/folder/on/server
```

for copying *towards* the server, or 

```bash
$ scp -rp amalia@saruman.bmt.tue.nl:/folder/on/server  /folder/on/your/pc 
```

Of course it helps if you replace `folder/on/server` and `/folder/on/your/pc` with actual paths to files or folders.



### (Re)moving files and folders

You can remove files using the `rm` command:

```bash
$ cd my_first_folder

$ ls
copy_of_file.txt    my_first_file.txt

$ rm my_first_file.txt

$ ls
copy_of_file.txt 
```

Likewise, you can remove folders using the `rm -r` command to do it recursively (removing a folder and the files/folders it contains).

Moving files and folders can be done using the `mv` command. This works similarly to the `cp` command (including the `-r` flag), naturally without copying:

```bash
$ ls
copy_of_file.txt    my_first_file.txt     my_first_folder

$ mv my_first_file.txt my_newly_named_file.txt

$ ls
copy_of_first_folder    my_first_$ ls
copy_of_file.txt    my_first_folder    my_newly_named_file.txt
```

**Be careful!** When moving a folder to a destination folder that does not exist yet, the folder's name will be changed instead. If the destination *does* exist, the folder will be *moved* to the destination folder, similarly to what the `cp` command does.

Like with the `cp` command, the `mv` command can move multiple files or folders to the same destination, for example:  `mv file1 folder1 file2 folder2 dest` will move the files and folders to the destination folder.


### File properties

You have probably noticed that a lot of interaction with Linux goes through using the `ls` and `cd` commands. The `ls` command can even be more useful if you use a few flags. For example, try to run the following in your home folder:

```bash
$ ls -lh

total 12
drwxrwxr-x 2 amalia amalia 4.0K Feb  8 16:54 copy_of_first_folder
drwxrwxr-x 2 amalia amalia 4.0K Feb  8 16:54 my_first_folder
drwxrwxr-x 2 amalia amalia 4.0K Feb  8 16:54 my_second_folder
```

This shows quite some information, that you don't need to understand fully at this point. To give an idea, let's break down what it means for `my_first_folder`.

* `drwxrwxr-x` are the permissions of the file or folder. In this case, the `d` shows you that it is a folder (folders are probably also given a distinct color depending on which terminal application you are using). For files, both the color and the `d` are absent.
* The permissions can be broken into three groups of three: in this case `rwx`, `rwx`, and `r-x`. 
    * The first group shows what the user that owns the file or folder (`amalia` in this case) can do with it. 
    * The second group shows you what the group that owns the file or folder can do (the group is also called `amalia`) with it.
    * The last group shows you what any other user can do with this file or folder.
* The permissions are always `r`, `w`, and/or `x`. 
    * `r` means that a user or group is able to read the file or folder.
    * `w` means that a user or group is able to write to the file or folder.
    * `x` means that a user or group can execute a file (i.e. run a script, run a program) or *access* (i.e. enter using `cd`) a folder.
* The `2` shows how many references there are to this file or folder. Linux can have links to files and folders that add to the total of references.
* `amalia` is the user that owns the file.
* The second `amalia` is the group that owns the file. Groups can be larger than one user, and are usually set by the administrator to give mulitple users access to a file. An example can be a dataset that only a set group of users are allowed to use.
* `4.0K` is the size of the folder **without its contents**. For files it shows the file size. For folders it is close to useless, because they are always `4.0K`. 
* `Feb  8 16:54` is the date the folder was last changed.

Surprisingly, `ls` or `ls -l(h)` does not show everything inside the folder, because some of it can be hidden. If you want to show all the hidden stuff in your home folder, include the `-a` flag, like this:

```bash
$ ls -lah

total 12
drwx------ 8 amalia amalia 4.0K Feb  8 14:02 .
drwxr-xr-x 8 root   root   4.0K Feb  8 09:30 ..
-rw------- 1 amalia   amalia   1.3K Feb  8 09:30 .bash_history
-rw-r--r-- 1 amalia   amalia    220 Feb  8 09:30 .bash_logout
-rw-r--r-- 1 amalia   amalia   3.8K Feb  8 09:30 .bashrc
drwxrwxr-x 2 amalia amalia 4.0K Feb  8 16:54 copy_of_first_folder
drwxrwxr-x 2 amalia amalia 4.0K Feb  8 16:54 my_first_folder
drwxrwxr-x 2 amalia amalia 4.0K Feb  8 16:54 my_second_folder
-rw-r--r-- 1 amalia   amalia    665 Feb  8 09:30 .profile
```

Your result may include other files or folders as well. The files starting with dots are invisible by default because they are configuration files. The `.bash_history` file contains the history of the commands you have typed. `.bashrc` and `.profile` contain some settings for your account. Normally you do not have to touch them, but it is good to know these files exist. If you look at the permissions, you can see that other users can only read the files, because of the `r--` in the 'other users' permissions. You will also see that `.` and `..` appear here as links which are followed when using `cd ..` to go to the folder one level-up, or when using `.` to refer to the current folder.


### Disk usage

As we have seen, the `ls -lh` command does not really help for establishing how much disk space a folder takes, because the folder's contents are not included. You can use the `du -sh` command instead. `du` stands for disk usage. The `-s` is there to list a total for a folder (otherwise it will list all files/folders in side it). The `-h` does the same as in the `ls -lh` command: it makes sure that the sizes are humanly readable (hence `-h`) and formatted to kilobytes, megabytes, and gigabytes.

```bash
$ du -sh my_first_folder
8.0K    my_first_folder
```

If you want to do this for multiple folders, you can simply add other file names. You can use a *wildcard* to include all files and folders in the current working directory:

```bash
$ du -sh *
8.0K    copy_of_first_folder
8.0K    my_first_folderi
8.0K    my_second_folder
```

Clearly, these folders are not that big: only 8.0 K or kilobytes. If you have large data sets, `du -sh` can display M or G to denote megabytes and gigabytes.

---

Tip
: The wildcards trick also works for the `ls`, `cp`, `mv`, and `rm` commands to list, copy, move, or remove multiple files. It also works if you want to match part of a name, i.e.

```
$ ls -lh *
8.0K    copy_of_first_folder
8.0K    my_first_folder
8.0K    my_second_folder

$ ls -lh *first*
8.0K    copy_of_first_folder
8.0K    my_first_folder

```

---


## Running programs and scripts

You now know how to get your data and code on the server, and how to organize it once you have done so. How to run that code? In this section, we are working with an Python script that you can save on the server. To run the examples, make a new folder in your home folder called `process_example`, and create a file called `my_clock.py`. Open the file in your favorite editor, and write the following code:

```python
import time
import datetime

while True:
    # create a datetime object for the current date and time
    now = datetime.datetime.now()

    # print the time in hours:minutes:seconds format
    print('{:02d}:{:02d}:{:02d}'.format(now.hour, now.minute, now.second))

    # wait a second
    time.sleep(1)
```

This script will print the time every second. Save the file. Check with `cat` or `more` if the file has been written.
Note that this script contains an infinite loop. It won't stop on its own.

Run the file with the command `$ python my_clock.py`. The script should print the current time every second.



### Quitting

The script is executed by the Python runtime. Technically, that is a program that runs the script for you. Linux will give this program a so-called process ID which you can use to identify it. Notice that the script is completly modal: you can't do anything else in the Terminal window until you quit it. Of course, you *can* always open a new Terminal window and make a connection to the server in another window to do other things.

If you want to stop the script from running, press <kbd>Ctrl-C</kbd>. This will result in a so-called `KeyboardInterrupt` exception in Python, which will automatically quit the Python runtime. It will also show during the execution of which line in the script it quit. Statistically, that would be the final line of the script, because the call to `time.sleep()` takes the most time.

---

Pro-tip
:   Running the `python` command with the `-i` flag runs the script in so-called 'interactive' mode. When you press <kbd>Ctrl-C</kbd>, the Python runtime will send a `KeyboardInterrupt`, but not quit (!). It returns to the Python interpreter (with a `>>>` prompt) with all classes, functions, variables still in memory. This is useful when debugging, because whenever your script crashes because of a bug, you can use the interpreter to assess what went wrong. You can try it out on the script above, and inspect what the value of the `now` variable is.

You will notice that the Python interpreter does respond to <kbd>Ctrl-C</kbd> by stopping *Python code*. To stop the interpreter at the prompt, press <kbd>Ctrl-D</kbd>. You can use <kbd>Ctrl-D</kbd> to quit any prompt: if you press it at the Bash prompt it will terminate the ssh-connection or close the Terminal window.

---

### Pausing

Sometimes it can be useful to pause a program temporarily, then do something else, and continue the program. You can press <kbd>Ctrl-Z</kbd> to pause virtually any program. It will put the current program in a sort of hibernation state, letting you do other things in the Terminal window. You can continue the program with the `fg` command. Note that the Terminal will print the process id of the current program when you pause it:

```
$ python -i my_clock.py
08:31:18
08:31:19
08:31:20
^Z
[1]  + 27152 suspended  python -i my_clock.py
```

In this case, the program id is 27152. When you issue the `fg` command, it will continue:

```
$ fg
[1]  + 27152 continued  python -i my_clock.py
08:32:21
08:32:22
08:32:23
```

### When <kbd>Ctrl-C</kbd> is not enough

If a script or program has crashed so badly it does not even respond to <kbd>Ctrl-C</kbd>, you can use <kbd>Ctrl-\\</kbd> which will stop *everything associated with a process*. This is particularly useful when your script uses multiple threads (e.g. using the `multiprocessing` library in Python).


## Running Python scripts on GPUs

If your script uses the GPU, for example through the Python libraries `tensorflow-gpu` or `theano`, it will automatically occupy **ALL GPUS**. If you are sharing the machine with other people (i.e. on a GPU-server), this may be the fastest way to get on their nerves. You need to contain your script to only use one (or at least a subset) of the available GPUs.

The first step is to check which GPUs are available. The BMT servers have a tool called `nvtop` to check this:

```bash
$ nvtop

 GPU 0: GeForce GTX TITAN X                                                    P8
 --------------------------------------------------------------------------------
   Util:               0%   Temp: ------       57C    Fan: --            22%
  Power: --                                     8%    19 / 250W
   VRAM:                                        1%    115 / 12204MiB
 --------------------------------------------------------------------------------
 User         PID   %CPU   %RAM   VRAM (MiB)   Command
 ariane       4052  66.3   17.3         1202   python experiment.py

 GPU 1: GeForce GTX TITAN X                                                    P8
 --------------------------------------------------------------------------------
   Util:               0%   Temp: ------       57C    Fan: --            22%
  Power: --                                     7%    17 / 250W
   VRAM:                                        2%    250 / 12207MiB
 --------------------------------------------------------------------------------
 User         PID   %CPU   %RAM   VRAM (MiB)   Command
 No processes

 GPU 2: TITAN Xp                                                               P2
 --------------------------------------------------------------------------------
   Util:               0%   Temp: -------      64C    Fan: ----          39%
  Power: ----------------                      45%    112 / 250W
   VRAM: --------------                        41%    4984 / 12189MiB
 --------------------------------------------------------------------------------
 User         PID   %CPU   %RAM   VRAM (MiB)   Command
 alexia       13733 80.6   19.3         4909   python script.py

--------------------------------------------------------- Press CTRL-C to quit ---
```

At the moment, GPUs 0 and 2 are occupied (indexes start at 0). As you can see, the tool prints the commands together with the user that is running it, the process ID, the amount of CPU and RAM it is using, and, most importantly, how much GPU memory is used. Currently, GPU 1 is doing nothing (as indicated by the `No processes` text), which means that there is nothing stopping you from using it.


### Running your script on a selected GPU in TensorFlow (with or without Keras)

In Keras and TensorFlow, you can select a GPU using the `CUDA_VISIBLE_DEVICES` environment variable. An environment variable is a variable in Linux that can be read by anything that runs in the current Terminal window. You can set these at the Bash prompt, like this:

```bash
$ CUDA_VISIBLE_DEVICES=1
```

**Do not blindly follow this example, but use the number of the GPU you actually can use.**

If you want to inspect the current setting, you can use `echo`, which is Bash's variant of the print statement. When you access environment variables you need to preprend them with a $-sign:

```bash
$ echo $CUDA_VISIBLE_DEVICES
1
```

Now, if you start to run a script that uses TensorFlow, it will run on GPU 1. You can test it out with the following example Python script:

```python
import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
with tf.Session() as sess:
    print('a: {}, b: {}'.format(sess.run(a), sess.run(b))
    print('Addition with constants: {}'.format(sess.run(a + b)))
    print('Multiplication with constants: {}'.format(sess.run(a * b)))
```

Run it with `python -i` to make sure that the Python interpreter keeps running after the script -- which only takes a few seconds to run -- stays alive. Pause the interpreter with <kbd>Ctrl-Z</kbd> and look at `nvtop` to see if you see if your script appears in the list for the GPU you selected. If it does not, you made a mistake somewhere. If the script appears on all GPUs, you have not set the `CUDA_VISIBLE_DEVICES` variable correctly.

Check how much GPU memory (VRAM) the script is using. It is probably using all it can get, which is clearly not necessary for a script that adds two numbers.

Quit `nvtop` with <kbd>Ctrl-C</kbd>. Press `fg` to go back to the Python interpreter, and quit that with <kbd>Ctrl-D</kbd>.


### Limiting memory use in TensorFlow (with or without Keras)

The GPUs in the `nvtop` example above have 12204, 12207, and 12189 megabytes of GPU memory available. Only part of it is used by the other scripts, which means there is room for more. It is best to first inquire with the associated users if it is OK to use these GPUs, as their scripts may use variable amounts of memory. As you saw before, by default TensorFlow just takes all GPU memory available on the GPU. **It is unlikely that you ever need this much memory. On a GPU server GPU memory is a precious resource, and it is rather impolite to use more than you need when sharing the server with other users.**

You can limit the amount of GPU memory your script uses in the script itself, by having these lines at the top:

* For TensorFlow only:
    ```python
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    ```

* For Keras and TensorFlow
    ```python
    import tensorflow as tf
    import keras.backend.tensorflow_backend
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.tensorflow_backend.set_session(session)
    ```

This sets the fraction of memory that is used to approximately 5%. You can then run additional scripts on the same GPU.


## Keep programs and scripts running after logging out

If you lose connection with the server, everything you were running in the Terminal on that server stops running. This is not ideal. You can use a so-called terminal multiplexer or `tmux` to keep your scripts running in the background. To do this, type `tmux`. You will see a green bar appear at the bottom of the Terminal window, indicating that you are using the multiplexer. You can run anything you like inside this multiplexer window. To test it out, you can use the my_clock script we used before.

To log out, you press <kbd>Ctrl-B</kbd> and then you press <kbd>D</kbd> to 'detach' from the current session. Anything you had running in the window will continue, even when you log out. When you want to go back to it, you can type the command `tmux -a` to 'attach' to the last session you had running. `tmux` is very powerful, allowing you to have multiple panels that run different things side-by-side. You can also have multiple sessions running in parallel, by detaching from the current session, and starting a new one by just typing `tmux` again. Use `tmux ls` to list all running sessions. With `tmux -a -t <session number>` you can attach to a specific session.

`tmux` has plenty more options that are very useful (for example, using multiple `tmux` windows or panes), but slightly outside the scope of this tutorial. Tutorials on using `tmux` are abundant on the internet however, so if you are interested have a look at them.


## Installing additional programs and packages

The availability of installation options will depend on your rights on the system you are using. We are assuming you are using Ubuntu here (which is installed on TU/e servers by default). You can install and update programs using the `apt` command. For example, if you want to install a medical image viewer, you can install ITKSnap with 

```bash
$ sudo apt install itksnap
```

`sudo` indicates you need an admin account, which you may have for your own PC but not for the server. In that case, inquire with the administrator how to proceed.

Python packages can be installed in your own user folder, for which you do not need an admin account. You can install them with the `pip` command. For example, to install TensorFlow 1.3.0, Keras 2.0.0, or SimpleITK you can run

```bash
$ pip install --user tensorflow-gpu==1.3.0

$ pip install --user keras==2.0.0

$ pip install --user simpleitk
```

To remove packages, use `$ pip uninstall <package_name>`.


## More on file permissions

In the [File properties](#file-properties) section, we have briefly looked at permissions. At some point, you may require to change these permissions on your own files, i.e. those for which `ls -l` shows your username. Those files, your username 'owns'. Only the administrator of a computer can change the ownership of the files. What you yourself can change however, are the permissions. You can set them using the `chmod` command. Remember that the permissions dictate if a file can be 1. read, 2. written to, or 3. executed by 1. users, 2. groups, or 3. everyone. 'Executed' here means 'run'. We have already seen that all folders are executable, which simply means 'accessible'.

To change the permissions on a file you own to forbid writing to it, simply type something like

```bash
$ chmod u-w some_filename
```

There `u` here refers to who the permissions are changed for, the `-` should be read as a 'minus' for 'removing', and the `w` for 'writing', i.e. this line says: 'Modify the permissions for the file `some_filename` such that the user that owns the file can no longer write to it'. Other options instead of `u` include `g` for group, `o` for others, and `a` for all. Other options instead of `w` are `r` for reading and `x` for executing. By using `+` you can add permissions, for example

```bash
$ chmod u+x some_filename
```

Let's you, the user (hence `u`), execute (hence `x`) the code in `some_filename`. This turns `some_filename` into a executable program.


## Running your own programs

Programs can be run using the `./` or `sh` commands. One way of writing them is using so-called shell scripts, which consist of the commands you have been typing on the prompt so-far. For example, if you save the code `cd ~; ls -l` to a file called `list_home.sh`, make it an executable by running `chmod u+x list_home.sh`, you can run it by typing `./list_home.sh`, which will (believe it or not) navigate to your home folder and list the files and folders in it. Note that the dot in `./` refers to the current folder.

Of course, you can write much more involved scripts in here, that truly are full programs (e.g. there are for loops and if statements available on the command line as well). That goes beyond this short tutorial, but if you search for 'shell scripts' you are bound to find some good tutorials on this subject.

You can do the same thing with Python scripts. That requires one extra comment be placed at the top of your Python file which tells it where to find the Python interpreter. First, we need to know where the Python interpreter is located on your system:

```bash
$ which python
/home/amalia/anaconda3/bin/python
```

Let's add that information to the my_clock script:

```python
#!/home/amalia/anaconda3/bin/python
import time
import datetime

while True:
    # create a datetime object for the current date and time
    now = datetime.datetime.now()

    # print the time in hours:minutes:seconds format
    print('{:02d}:{:02d}:{:02d}'.format(now.hour, now.minute, now.second))

    # wait a second
    time.sleep(1)
```

Now, make the script executable:

```bash
$ chmod u+x my_clock.py
```

And run it with

```bash
$ ./my_clock.py
14:51:09
14:51:10
14:51:11
```

## The PATH variable and ~/.bashrc file

If the `./` is bothering you, here is the explanation for it: Linux has a lot of scripts and programs that you can run without that `./`, examples include `tmux`, and `python`, and `nvtop`, and `htop`, and even `du`. The reason you do not need the `./` there, is because these programs are in your PATH. The PATH variable is a Linux variable that lists all locations in which Linux is looking for executables when you try to run them without `./`. By requiring that you use `./` for scripts that are not on the path, Linux makes sure you cannot overwrite the default ones. For example, if you have an executable file called `du` in your folder, how should Linux know which `du` to run? The program that shows disk usage? Or this obscure file you wrote?

The answer is, it is using that version that it can find in the PATH variable. To inspect the PATH variable on your computer, you can type

```bash
$ echo $PATH
/home/amalia/bin:/home/amalia/.local/bin:/home/amalia/anaconda3/bin:/usr/local/cuda-8.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
```

The locations are separated by colons. As you can see, one of the locations is the `anaconda3/bin` folder. We have seen in the previous section that that is where we can find the Python executable. This is the reason why Linux knows where to look when you type `python` on the command line. The `du` program is in `/usr/bin/` which is also part of the PATH.

If you want, you can add folders to this path. For example, if you have a folder `scripts` on your home folder that contains some executable scripts like `my_clock.py`, you can add that folder to PATH by typing

```bash
$ export PATH="/home/amalia/scripts:$PATH"
```

Here, we are basically redefining PATH by appending the `scripts` folder. Now, you can type

```
$ my_clock
14:57:29
14:57:30
14:57:31
```

Congratulations are in order, you just made a fully functioning Linux program! However, if you logout and back in, you will notice the PATH file has been reset. You can change this by adding the line `export PATH="/home/amalia/scripts:$PATH` to the `.bashrc` file in your home folder. This file will be run every time you log in, and the PATH variable will be updated automatically. If you look around in the `.bashrc` file, you will see that this file also contains an `export PATH=` statement for the `anaconda` installation.

## Extras

There are plenty more Linux commands that can be very useful, but it goes too far to list them all here. A few of the more useful ones are listed below. These can be convenient, but they are in no way required for casual Linux use.

* `man` for manual let's you view a command's documentation. For example, type `man ls` to view all options for the `ls` command. You can navigate with the arrow keys and page up/down keys. Quit the manual by typing `q`.

* `top` (**t**able **o**f **p**rocesses) or `htop` (Hisham's version of top) is the Linux equivalent of the task manager or activity monitor. Here, all process are listed with the users that use them, their process ID (PID), and their CPU and memory usage.

* `less` is a more advanced alternative to `more` that also let's you navigate with the page up/down keys. There are more options, like searching, which have the same commands as the `vim` editor.

* `grep` (**g**lobally search a **r**egular **e**xpression and **p**rint) to find text in files, for example `grep import my_clock.py` finds the lines where an import statement is used.

* `>` lets you divert the output of a command to a file, for example `grep import my_clock.py > test.txt` will put the import statements in a new file called `test.txt`. This can be useful when your script outputs a lot of logging information that you wish to save.

* `tail` prints the final lines of a file. You can set how many lines are shown with the `-n <number>` flag. If you use the `-f` flag, it will update the view with recent additions to the file, which is ideal for log files created with `>`. For example, if you use `python my_clock.py > times.txt` and view the file in another Terminal or `tmux` instance with `tail -f times.txt`, it will update the window with the recently printed times.

* `head` behaves like `tail` but then for the first `-n` lines. No `-f` flag is available however.

* `|` lets you divert output of one command to another command, for example, you can chain `grep` commands like this: `grep import my_clock.py | grep date` to only show lines that contain `import` and `date`.

* `history` lets you view all commands you have typed. You can use it with `grep` to search in it, by piping the output, like this: `history | grep python`.



