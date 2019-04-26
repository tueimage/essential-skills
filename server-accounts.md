## Connecting to a Linux server

First, make sure that you can get an account on the server. Contact your supervisor for the specifics. 
If you use Windows, you will need to install a Terminal application called Cygwin first.


### Windows only: installing Cygwin

1. Download Cygwin from https://cygwin.com/install.html. Most likely you will need the 64-bit version.
2. Open the installer.
3. Choose next on the first screen.
4. Choose ‘Install from Internet’ when selecting an Installation Type. Click next.
5. You can leave the installation folder to the default option. Click next.
6. On the next screen you can install packages in Cygwin. Select the checkbox next to openssh (under ‘Net’) and xinit and xorg-server (under ‘X11’). Click next.
7. Leave ‘Select required packages’ on and click next again. The installation will start.
8. After the installation has completed, you can optionally let the installer create icons on the Desktop and Start Menu.


### Getting the keys to the server

You are going to setup a secure key that you can use to login to the server. A key is basically a super long password that will be used to let the server know that it is really you who wants access. The key consists of two parts: a public key and a private key. The private key is on your own computer. The public key will be put on the server by the administrator. 

When you log in, your computer will send an encrypted version of the private key to the server, that will be validated against the public key on the server. If they match, you will get access. To generate a key, follow these steps:

1. Type `ssh-keygen -t rsa` on the prompt and press enter.
3. You will be asked for a location to save the key. The default is fine, so just press Enter, but in any case write down the path.
4. Enter a passphrase (a password used to open the key), and write it down so you don't forget it.
5. Re-enter the passphrase to confirm.
6. Open the `id_rsa.pub` file where the public key was saved (you wrote down the location in step 3).
7. Paste the contents of the `id_rsa.pub` file containing the public key in an email to the administrator of the server. It should start with `ssh-rsa AAAA` and end your *local* (i.e. your PC) username and computer name.
8. Include the name of your supervisor in the email to the administrator. If you are going to use GPUs, also answer the following questions:
	1. How are you going to make sure that your scripts will only use **one GPU**.
	2. How are you going to limit the **amount of GPU memory** that your script will use. 
9. If you do not know the answers to these two questions, read the part of the [Linux tutorial about running scripts on GPUs](linux-essentials.md/#running-python scripts-on-gpus).

The administrator will now make sure the key will be known by the server. Once you get a confirmation that this is done, you can login to the server.


### Logging in

From the prompt, type the following command:

```
ssh -X amalia@saruman.bmt.tue.nl
```

This will let the user `amalia` login to the server at the address `saruman.bmt.tue.nl` using the secure shell. Of course, you will have to replace this with your name and the correct address for the server you want to use. The `-X` is there to make sure you can view plots and figures on the server later (see `Running Python on the server`).

You will now see some text that indicates your login was succesful. It will also display which machine you are connected to and which operating system it uses. Depending on the operating system, there may be a  more text that you can ignore for now.

Below all that text is a new prompt. By default, it will look something like this:

```bash
amalia@saruman:~$
```

From this prompt you can type commands like `whoami` or `date`. Now, these commands will run *on the server*. The results will be send back to your own PC and show up on the screen like normal. It is virtually indistinguishable from running commands on your own system, except for the difference in prompt.

To quit the connection and log out, press <kbd>Ctrl-D</kbd>. This will return you to the prompt on your own PC.
