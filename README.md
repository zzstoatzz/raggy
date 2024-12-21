# raggy

a Python library for scraping and document processing

## installation

```python
pip install raggy
```

add extras to load different document types:
```python
pip install raggy[chroma]     # ChromaDB support
pip install raggy[tpuf]       # TurboPuffer support
pip install raggy[pdf]        # PDF processing
```

read the [docs](https://zzstoatzz.github.io/raggy/)

### what is it?

a simple-to-use Python library for:

- scraping the web to produce rich documents
- putting these documents in vectorstores
- querying the vectorstores to find documents similar to a query

> [!TIP]
> See this [example](https://github.com/zzstoatzz/raggy/blob/main/examples/chat_with_X/website.py) to chat with any website, or this [example](https://github.com/zzstoatzz/raggy/blob/main/examples/chat_with_X/repo.py) to chat with any GitHub repo.

### license and dependencies

> [!IMPORTANT]
> This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

### contributing

I would welcome contributions! See the [contributing guide](https://zzstoatzz.github.io/raggy/contributing) for details.
