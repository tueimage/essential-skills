# Version control with Git

Git is a version control system. You can use Git on any folder to turn it into
a *repository* with version control. This means you can make changes in the folder and
*commit* them to a new version of that folder. If you later regret the changes, or want to figure out how that nasty bug got into your code, you can *check out* previous versions of the folder, and
 *revert* to them.

In addition to this, Git allows collaboration within a repository. You can make separate branches in which teams can work on different functionality. These branches are complete versions of the repository to which changes can be made. Each branch will have their own versioning history. Once the implementation of a new function is done and tested, the branches can be merged again. 

This tutorial is split into three parts: one part focussing on version control within a local repository, one part that focuses on branching and mergine, and one on collaborating using GitHub.

Git is a command line tool. Although a plethora of GUI-based applications for interaction with Git exist, in this tutorial we are going to stick with the command line interface, as it is the most universal way to interact with Git: it will even work on remote computers, like computational servers, over SSH. Like in the Linux chapter, we are using the convention that the '$'-sign indicates a command line prompt, and that any command you type should not include this sign.



## Local repositories

#### Turning a folder into a Git repository

Fire up your terminal application (e.g. Cygwin on Windows) and navigate to the folder you want to place the repository in, for example your documents folder. If you do not know how to do this, we refer you to the [Linux tutorial](linux-essentials.md).

Once in the correct folder, make a new folder called `my_repository` and navigate to it by typing

```
$ mkdir my_repository
$ cd my_repository
```

Now let's turn this folder into a repository, by typing

```
$ git init
```

If Git is installed correctly, it will respond with something like

```
Methode Initialized empty Git repository in /some/long/path/my_repository/.git/
```

Now, to check the repository's status, you can type

```
$ git status
```

which will say

```
On branch master

No commits yet

nothing to commit (create/copy files and use "git add" to track)
```

This shows that you there have not been any changes to the repository yet (you have added no files or folders yet), and that you are currently on the branch called 'master'. For now, you can ignore any information on branches until the second part of this chapter.

During this chapter, we will use `git status` frequently to keep track of what is happening with our repository.



#### Committing versions

Let's add a file to our repository called `vector.py`, which will contain a Python class for vectors. Open the file, copy-paste the following code:

```python
class Vector:
    def __init__(self, *elements):
        self.elements = elements
```

and save the file.

If you now type `git status`, the following message should be shown:

```
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)

        vector.py

nothing added to commit but untracked files present (use "git add" to track)

```

Git has noticed you have added a file to the repository. However, it reports
the files is not tracked yet, that is, Git does not save versions of this file.
(Also, if you push this repository to GitHub, the file will not be included, as we will see later.)

To let Git track the file, type 

```
$ git add vector.py
```

If you now again look at the status, it will print

```bash
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)

        new file:   vector.py

```

Git tells you that, should you commit a new version, the addition of the new file `vector.py` will be part of the commit. Let's do just that. Type

```bash
$ git commit -m 'Added vector.py'
```

The `-m` flag lets you include a message for this commit, that you should use to document what you did since the previous commit. If we now again check `git status`, it prints

```bash
On branch master
nothing to commit, working tree clean
```

Your working tree is clean, which means: no new files have been added that are not tracked, and no new changes to the already existing files have been made. Git keeps a log of all the commits that you have made. Type 

```bash
$ git log
```
 
too see this:

```bash
commit 7fa5007bc326ff8a4bf78912f41a21130d8165b9 (HEAD -> master)
Author: Koen Eppenhof <k.a.j.eppenhof@tue.nl>
Date:   Wed Apr 10 13:05:31 2019 +0200

    Added vector.py
(END)
```


#### Committing more changes

Let's change the `vector.py` by adding a `__repr__()` method to the class:

```python
class Vector:
    def __init__(self, *elements):
        self.elements = elements

    def __repr__(self):
        s = '['
        for x in self.elements:
            s += str(x) + ', '
        s += ']'
        return s
```

If you look at the status now, it says that the file has changed:

```bash
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   vector.py

no changes added to commit (use "git add" and/or "git commit -a")

```

We can commit this change as well by typing 

```bash
$ git add vector.py
$ git commit -m 'Added __repr__() method'
```

You first need to 'stage' the changed files you want to commit with `git add` before committing them. If you have many changed files, this becomes quite cumbersome, which is why you can type

```bash
$ git commit -a -m 'Added __repr__() method'
``` 

instead, where `-a` stands for 'all changed files'. You can again use `git log` to see the history of commits.

When adding files you will still need to explicitly use `git add` however. When removing files, there is the equivalent `git rm`. This will both remove the file itself, as well as stop the version tracking of it:

```bash
$ git rm vector.py
```

If you now type `ls`, no files will be shown. Let's first commit this new change, then worry about how to get that file back:

```bash
$ git commit -a -m 'Deleted vector.py'
```


#### Reverting to previous versions

To revert the commit in which we deleted the file, we want to go back to go back to the state of the repository one commit before that. In Git, we can do this with with the `git revert` command. There are two ways of specifying to which commit you want to return: by specifying the identifier of the commit that is shown in `git log` or by using relative refererences.

##### Reverting by identifier

In the `git log` you can find the identifier. In this case it is 52e352fbb0caf74c631c1054da1e6dcd4c690786.

```
  commit 52e352fbb0caf74c631c1054da1e6dcd4c690786
  Author: Koen Eppenhof <k.a.j.eppenhof@tue.nl>
  Date:   Wed Apr 10 14:11:36 2019 +0200

      Deleted vector.py

  commit 907ae2e2338cc302adca23d17cf2cfc72ed623d4
  Author: Koen Eppenhof <k.a.j.eppenhof@tue.nl>
  Date:   Wed Apr 10 14:00:18 2019 +0200

      Added __repr__() method

  commit 7fa5007bc326ff8a4bf78912f41a21130d8165b9
  Author: Koen Eppenhof <k.a.j.eppenhof@tue.nl>
  Date:   Wed Apr 10 13:05:31 2019 +0200

      Added vector.py
```

Luckily you do not have to type in the whole thing. Only the first eight characters suffice, but if you type only the first few and press <kbd>Tab</kbd> it will auto-complete.

```bash
$ git revert 52e352f
```

This will open an editor in which you can type a commit message, although a 
default message reading `Revert "Deleted vector.py"` is provided there already.

Upon quitting the editor, the reversion is committed, and the file `vector.py` should be back in the folder.


##### Reverting by relative refererence

If you know you how many commits you want to go back, you can use a relative
reference when reverting. You can specify a reference relative to the current HEAD. HEAD is a tag that is always attached to the current state of the repository.
If you want to undo the last commit, you want to go back two commits, by typing this:

```
$ git revert HEAD~2
```

This will have the same effect as reverting by identifier.


#### Checking out commits

Finally, there is one more useful command that you can use to check out the status of the repository in a commit. Using `git checkout <IDENTIFIER>` or `git checkout <RELATIVE REFERENCE>` to set the state of the folder to that commit. You can use this to check out the contents of files in that commit.

```
$ git checkout HEAD~1
```

You can go back to the last commit by typing

```
$ git checkout master
```

(At least, assuming you have not changed branches, more on that later.)
Be careful though. If you make changes in a checked out commit, these will not be saved, unless you commit them, and *merge* them. That is possible, but is beyond the scope of this tutorial.


## Branching and merging

#### Branching
Git branch
Git checkout

#### Merging

Merge conflicts




## Collaborating on Github

Git push
Git pull
