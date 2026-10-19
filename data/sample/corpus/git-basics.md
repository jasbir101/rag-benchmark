# Git basics: commits, branches and merges

Git is a distributed version control system. Every clone of a repository contains the full history, so most operations, such as viewing logs or comparing versions, work offline.

Changes move through three areas. The working tree holds the files you edit. The staging area, also called the index, holds the changes you have selected for the next commit with `git add`. A commit, created with `git commit`, records a snapshot of the staged content together with an author, a timestamp and a message. Each commit is identified by a hash of its contents and its parent commits.

A branch is simply a movable pointer to a commit. Creating a branch is cheap, which encourages working on each feature or fix in isolation. `git switch -c feature-name` creates a branch and moves to it.

When work on a branch is done, it is combined with another branch. A merge creates a new commit with two parents and preserves the exact history of both lines of development. If the target branch has not moved since the feature branch was created, Git performs a fast-forward instead and simply advances the pointer.

Rebasing is an alternative that replays the feature branch's commits on top of the target branch, producing a linear history. Because rebasing rewrites commits, it should not be used on commits that other people have already pulled.

A merge conflict happens when both branches changed the same lines. Git marks the conflicting region in the file with conflict markers; you edit the file to the intended result, stage it and complete the merge with a commit.

To share work, `git push` uploads local commits to a remote repository and `git pull` fetches remote commits and integrates them into the current branch.
