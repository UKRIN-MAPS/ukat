In order to generate/update the HTML documentation page, it's required to install the following:
`pip install -U sphinx`
`pip install sphinx_rtf_theme`

Then, open a terminal and type the following commands within the ukat parent folder:
`cd doc`
`sphinx-apidoc -M -f -e -t ./_templates -o . ../ukat`
`make html`

For more information, watch the following [Youtube video](https://www.youtube.com/watch?v=b4iFyrLQQh4)

