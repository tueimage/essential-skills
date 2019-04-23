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
Initialized empty Git repository in /some/long/path/my_repository/.git/
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


#### Undoing the last commit

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

Luckily you do not have to type in the whole thing. Only the first eight characters suffice, but, even better, if you type only the first few and press <kbd>Tab</kbd> it will auto-complete.

```bash
$ git revert 52e352f
```

This will open an editor in which you can type a commit message, although a 
default message reading `Revert "Deleted vector.py"` is provided there already.

Upon quitting the editor, the reversion is committed, and the file `vector.py` should be back in the folder.


##### Reverting by relative refererence

If you know you how many commits you want to go back, you can use a relative
reference when reverting. You can specify a reference relative to the current HEAD. HEAD is a tag that is always attached to the last commit of the current branch (more on branches later).
If you want to undo the last commit, you revert the commit tagged HEAD, by typing this:

```
$ git revert HEAD
```

This will have the same effect as reverting by identifier.


#### Undoing the N previous commits

**This should only be done on local repositories or branches, i.e. branches that are on your own computer.**

You can undo multiple commits using `git reset --hard`. You specify the commit you want to return to using an identifier or a relative reference. For example, you can type

```
$ git reset --hard 907ae2e2
```

or

```
$ git reset --hard HEAD~2
```

to revert two commits.


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

###### Exercises

* Add a method `__len__()` to the `Vector` class:
    ```python
        def __len__(self):
            return len(self.elements)
    ```
    Commit the change and add an appropriate message.
    
    <details><summary>Answer</summary><p>
    `git commit -a -m 'Added __len__() method'
    </p></details>

* Add a file called `test.py` with the following content:
    ```python
    from .vector import Vector
    v1 = Vector(1, 2, 3)
    v2 = Vector(3, 2, 1)
    print(v1)
    print(len(v2))
    print(v1 + v2)
    ```

    And commit this change to the repository

    <details><summary>Answer</summary><p>
    ```bash
    git add test.py
    git commit -a -m 'Added test module'
    ```
    </p></details>

* Delete the `test.py` and commit this change to the repository
    <details><summary>Answer</summary><p>
    ```bash
    git rm test.py
    git commit -a -m 'Removed test module'
    ```
    </p></details>

* Revert to the previous commit to restore the test module
    <details><summary>Answer</summary><p>
    ```bash
    git revert HEAD~1
    ```
    </p></details>


## Branching and merging

Consider the following situation: you have a perfectly working folder full of wonderful code. Now, someone asks you to add some extra complex functionality to your code. This new code might break your perfect code, introduce bugs, or otherwise disturb your quiet life. Of course, Git allows you to go back in time to when things were working fine, but that would also remove all the work you did on the new functionality.

In this case, it is best to make a new *branch* in your repository.
Whenever you make a branch, the history of the repository splits in two.You can make commits in the new branch without it affecting the history or code in your main *master* branch.

The way this is often used is to have a stable master branch that contains well-tested code, and a development branch that has new functionality that has not been fully-tested yet. Once the new code *is* fully tested, you can merge the changes in the development branch into the master branch.


#### Branching

Creating and switching branches can be done using the `git checkout` command. In fact, switching branches is a little similar to switching to previous commits. Let's create a new branch called `addition`:

```bash
git checkout -b 'addition'
```

Git will respond with `Switched to a new branch 'addition'`. In this branch you can make changes to the code. For example, we can add an `__add__()` method to our `Vector` class, that returns the number of elements:

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

    def __len__(self):
        return len(self.elements)

    def __add__(self, other):
        assert len(self) == len(other)
        sums = []
        for a, b in zip(self.elements, other.elements):
            sums.append(a + b)
        return Vector(*sums)
```

Now, we commit this change:

```bash
git commit -a -m 'Added __add__() method'
```

Remember that this change is only reflected in the `addition` branch we are in. The `master` branch has not had the same update. Let's check that out:

```bash
$ git checkout master
Switched to branch master
$ more vector.py
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


#### Merging

Let's assume the new `__add__()` method has been extensively tested, and it works perfectly fine. We can now merge the feature branch `addition` into the master branch. This is done by first checking out the `master` branch if you haven't done so already,

```bash
$ git checkout master
Already on 'master'
```

and typing

```bash
$ git merge addition
```

#### Merge conflicts

Say you have made two new additional features to the `Vector` class, each in their own branch from master. Let's call the branches `feature1` and `feature2`, like this:

```
$ git checkout master
$ git checkout -b feature1 
Switched to branch 'feature1'

        ... adding feature 1 ...

$ git commit -a -m commit 'Added feature 1'
[feature1 b016669] 2
 1 file changed, 1 insertion(+)

$ git checkout master
Switched to branch 'master'

$ git checkout -b feature2
Switched to branch 'feature2'

        ... adding feature 2 ...

$ git commit -a -m commit 'Added feature 2'
[feature2 b016669] 2
 1 file changed, 1 insertion(+)

```

Now you want to merge both into master, so you checkout master, and merge the first feature:

```
$ git checkout master
Switched to branch 'master'

$ git merge feature1
Updating 4e26bf7..b016669
Fast-forward
 vector.py | 1 +
 1 file changed, 1 insertion(+)
```

So far so good. Now, we also merge `feature2`:

```
$ git merge feature2
```

This will result in a warning:

```
Auto-merging vector.py
CONFLICT (content): Merge conflict in vector.py
Automatic merge failed; fix conflicts and then commit the result.
```

The reason for this is that both the `master` branch and the `feauter2` branch are changed after the branching point, and both have changes to the same file(s). This results in Git not knowing which of the two existing versions is the 'truth'. Should the result have feature1 but not feature2, or feature2 but not feature1, or both, or neither?

This is called a *merge conflict*, and they are particularly abundant when collaborating. Luckily, merge conflicts are easy to solve. If you open the `vector.py` file you will see that Git has moved both features in the file, and you get to pick which version you want by removing text. For example, if the `__len__` and `__add__()` were added in `feature1` and `feature2` respectively, the file could look like this:

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

<<<<<<< HEAD
    def __len__(self):
        return len(self.elements)

=======
    def __add__(self, other):
        assert len(self) == len(other)
        sums = []
        for a, b in zip(self.elements, other.elements):
            sums.append(a + b)
        return Vector(*sums)
>>>>>>> feature2
```

The part between <<<<<<< and >>>>>>> is different in the `master` and `feature2` branches. The part above the `=======` is in `master`, the part below in `feature2`. In this case, you can resolve the conflict by simply removing the lines with `<<<<<<<< HEAD`, `=======`, and `>>>>>>> feature2` and saving the file. Then a new commit will close the conflict definitively:

```
$ git commit -a -m 'Merged feature2 into master and solved merge conflict.'
```

###### Exercises

* Create a new branch with an appropriate name and add the following method to the `Vector` class:

    ```python
        def __abs__(self):
            return Vector(*[abs(x) for x in self.elements])   
    ```
    Commit the change and add an appropriate message.
    
    <details><summary>Answer</summary><p>
    `$ git checkout -b dev`
    `$ git commit -a -m 'Added __abs__() method to Vector class'`
    </p></details>

* Go back to the `master` branch and add the following method to the `Vector` class:

    ```python
        def __abs__(self):
            c = 0
            for x in self.elements:
                c += x ** 2
            return c ** 0.5
    ```

    Commit the change and add an appropriate message.

    <details><summary>Answer</summary><p>
    `$ git checkout -b master`
    `$ git commit -a -m 'Added __abs__() method to Vector class'`
    </p></details>

* Merge the first branch into the `master` branch and solve the merge conflict.
    
    <details><summary>Answer</summary><p>
    `$ git merge dev`

    will result in a merge conflict that can be solved by opening the `vector.py` file, and removing the version of the `__abs__()` method you don't like. Then, commit the new version, like this:

    `$ git commit -a -m 'Merged dev branch and solved merge conflict.'`   
    </p></details>


## Collaborating on Github

Git push
Git pull
