# Live docs
A live version of latest documentation is hosted at http://serrano-lib.org/docs/

# Generate local docs
1. Install [mkdocs](http://www.mkdocs.org/) to generate guides and tutorials
```bash
$ pip install mkdocs
```
2. Install [realm/jazzy](https://github.com/realm/jazzy) to generate APIs and classes reference
```bash
$ [sudo] gem install jazzy
```
2. Clone repo and generate docs
```bash
$ git clone https://github.com/pcpLiu/serrano
$ cd serrano/
$ chmod u+x docs/doc_generate.sh
$ ./docs/doc_generate.sh 
```

The generated docs are located in `docs/` folder.