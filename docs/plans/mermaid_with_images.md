https://stackoverflow.com/questions/72585185/is-there-a-way-to-use-a-local-image-file-in-a-mermaid-flowchart

I want to try prototyping getting mermaid diragams running locally that I can render to png but we are leveraging the prebaked icon images from the `diagrams` python package which puts them in:

`.venv/lib/python3.12/site-packages/resources/**/*.png`

Even though their code is here:

`.venv/lib/python3.12/site-packages/diagrams`

It was an odd choice in how they layout their package by not keeping their shit together 🤷🏻‍♂️