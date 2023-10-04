# Cook2LTL: Translating Cooking Recipes to LTL Formulae using Large Language Models

This repository contains code and technical details for the paper:

**[Cook2LTL: Translating Cooking Recipes to LTL Formulae using Large Language Models](https://arxiv.org/abs/2310.00163)**
<p align="center">
  <img src="/images/cover.jpeg" alt="Cook2LTL Cover" />
</p>
Authors: Angelos Mavrogiannis, Christoforos Mavrogiannis, Yiannis Aloimonos

Please cite our work if you found it useful:
```
@article{mavrogiannis2023cook2ltl,
      title={Cook2LTL: Translating Cooking Recipes to LTL Formulae using Large Language Models}, 
      author={Angelos Mavrogiannis and Christoforos Mavrogiannis and Yiannis Aloimonos},
      journal={arXiv preprint arXiv:2310.00163},
      year={2023},
}
```
# Overview
Cook2LTL is a system that receives a cooking recipe in natural language form, reduces high-level cooking actions to robot-executable primitive actions through the use of LLMs, and produces unambiguous task specifications written in the form of LTL formulae.

<p align="center">
  <img src="/images/pipeline.jpeg" alt="Cook2LTL Pipeline" />
</p>

## System Architecture
The input instruction $r_i$ is first preprocessed and then passed to the semantic parser, which extracts meaningful chunks corresponding to the categories $\mathcal{C}$ and constructs a function representation $\mathtt{a}$ for each detected action. If $\mathtt{a}$ is part of the action library $\mathbb{A}$, then the LTL translator infers the final LTL formula $\phi$. Otherwise, the action is reduced to a lower-level of admissible actions $\{a_1,a_2,\dots a_k\}$ from $\mathcal{A}$, saving the reduction policy to $\mathbb{A}$ for future use. The LTL translator yields the final LTL formulae from the derived actions.


# Named Entity Recognition

The first step of Cook2LTL is a semantic parser that extracts salient categories towards building a function representation $\mathtt{a}=\mathtt{Verb(What?,Where?,How?,Time,Temperature)}$ for every detected action. To this end, we annotate a subset (100 [recipes](brat/recipes)) of the [Recipe1M+ dataset](http://pic2recipe.csail.mit.edu) and fine-tune a pre-trained spaCy NER BERT model to predict these categories in unseen recipes at inference. To annotate the recipes we used the [Brat annotation tool](https://brat.nlplab.org) ([installation instructions](https://brat.nlplab.org/installation.html)). To fill in context-implicit parts of speech (POSs), we use ChatGPT or manually inject them into the sentence. The resulting edited recipes can be found in the [edited_recipes.csv](edited_recipes.csv) file. The code to train the NER model is in `train_ner.py`.
<p align="center">
  <img src="/images/annotation.jpeg" alt="Annotated recipe steps" />
</p>

# Reduction to Primitive Actions
<p align="center">
  <img src="/images/action_reduction.jpeg" alt="LLM action reduction example" />
</p>

## Action Reduction
If $\mathtt{a}\notin\mathcal{A}$, we prompt an LLM with a pythonic import of the set of admissible primitive actions $\mathcal{A}$ in the environment of our use case, two example function implementations using actions from this set, and the function representation $\mathtt{a}$ in the form of the pythonic function along with its parameters. This is implemented in the `action_reduction` function by calling `gpt-3.5-turbo` through the OpenAI API. The new action is added to the pythonic import, enabling the system to invoke it in future action reductions.
> Note: An OpenAI API key is required for. You can create an account and set up API access [here](https://openai.com/blog/openai-api).

## Action Library
Every time that we query the LLM for action reduction, we cache $\mathtt{a}$ and its action reduction policy (in the `cache_action` function) to an action library $\mathbb{A}$ for future use through a dictionary lookup manner. At runtime, a new action $\mathtt{a}$ is now checked against both $\mathcal{A}$ and $\mathbb{A}$. If $\mathtt{a}\in\mathbb{A}$, which means that $\mathtt{Verb}$ from $\mathtt{a}$ matches the $\mathtt{Verb}$ from an action $\mathtt{a}_{\mathbb{A}}$ in $\mathbb{A}$ and the parameter types match (e.g. both actions have $\mathtt{What?}$ and $\mathtt{Where?}$ as parameters), then $\mathtt{a}$ is replaced by $\mathtt{a}_\mathbb{A}$ (`reuse_cached_action` function) and its LLM-derived sub-action policy, and passed to the subsequent LTL Translation step.
