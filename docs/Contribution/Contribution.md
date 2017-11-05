# Contribution Guide 

Please contribute to improve Serrano :hot_pepper: helping make it better.
Before opening an issue or submitting a pull request, please read through this guide.

## Issue

#### Bugs

In repository [pcpLiu/Serrano](https://github.com/pcpLiu/Serrano), only __Bug__ issues are accepted.
And please follow this [Issue template](https://github.com/pcpLiu/Serrano/blob/master/.github/ISSUE_TEMPLATE.md) to submit.

#### Other issues

All other issues, feature request, questions and general discussion. You can open issues at [pcpLiu/SerranoExplore](https://github.com/pcpLiu/SerranoExplore).


## Pull Request

We are trying to keep it easy to contribute to Serrano.
There are a few guidelines that you should follow so the community can keep track of changes.

#### Getting started

- __Submit an issue before you making a PR__. As addressed before, the issue should be placed in the correct repository.
- __Writing code__. 
	- __Read__ [__Coding Guidelines.__](/Contribution/Contribution/#coding-guidelines)
	- __Commit message__. Write meaningful and formated commit messages. Below is an example:
		
		> Fixed #3223.
		>
		> The first line should be a brief statement describing what you done and if applicable should also include an issue number, prefixed with its hash.

- __Pull Request__
	- __Fill PR template :clipboard:__. When you open a pull request, please fill the PR template displayed  
	- __Keep history clean :wastebasket:__. Please squash all your commits and rebase them properly making the git logs clean and easy to understand.
	- __Add your name to CONTRIBUTORS.md :pencil:__. Don't forget to add your names into CONTRIBUTORS.md 

!!! note "Docs Pull Request":
	For simple errors like typos or format issues in documentations, you can directly make a PR. You don't need to open an issue.


#### Coding Guidelines

We don't have very strict coding guidelines.
Just few basic principles to follow:

- __Comments are required__. 
	- __All classes, functions, structs etc. require header comment__. Serrano automatically generates API reference via [jazzy](https://github.com/realm/jazzy). You can check existing code to see how to write header comments, basically they are just Markdown.
	- __Details are always better__. If you are not sure if need to write down some comments here so that people could understand it, maybe you need to write it. Make sure the community could understand your logic quickly via the help of your comments.
- __Test code are required__. Please add test cases or create a new test file according to your situation.
- __Code style__. There's no strict code style. Just make sure your code conforms to the style around the whole project.






