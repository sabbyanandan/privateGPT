# privateGPT

This repo is a fork of https://github.com/imartinez/privateGPT, and all the things are inherited as-is from upstream with subtle changes, including lazy instantiation of GPT to bridge the ongoing user interaction through a statically served minimalistic UI.

The `source_documents` folder is ignored from the commit; hence it is not in this repo, and that needs to exist with the source documents to fulfill the initial ingestion and indexing.

Follow the steps described in the upstream repo to set up the environment and dependencies, and after the ingestion step is complete, run `python privateGPT.py` to bootstrap the web server at http://localhost:5000/app