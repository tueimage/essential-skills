# Version control with Git

Git is a version control system. You can use Git on any folder to turn it into
a *repository* with version control. This means you can make changes in the folder and
*commit* them to a new version of that folder. If you later regret the changes, or want to figure out how that nasty bug got into your code, you can *check out* previous versions of the folder, en *revert* to them.

In addition to this, Git allows collaboration within a repository. You can make separate branches in which teams can work on different functionality. These branches are complete versions of the repository to which changes can be made. Each branch will have their own versioning history. Once the implementation of a new function is done and tested, the branches can be merged again. 

This tutorial is split into three parts: one part focussing on version control within a local repository, one part that focuses on branching and mergine, and one on collaborating using GitHub.

Git is a command line tool. Although a plethora of GUI-based applications for interaction with Git exist, in this tutorial we are going to stick with the command line interface, as it is the most universal way to interact with Git: it will even work on remote computers, like computational servers, over SSH.



## Local repositories

#### Turning a folder into a Git repository

Git init
Git status
Git log

#### Committing versions

Git add
Git commit
Git status
Git log
Git rm

#### Checking out versions

Git checkout VERSION
Git status

#### Reverting to previous versions

Git revert
Git reset



## Branching and merging

#### Branching
Git branch
Git checkout

#### Merging

Merge conflicts




## Collaborating on Github

Git push
Git pull
