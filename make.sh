echo Creating PDF version
pandoc metadata.yaml sotw_de.md -o Solo_der_Woche.pdf --from markdown -V lang=de-DE  -V geometry:margin=3cm -V papersize=a4paper -fmarkdown-implicit_figures