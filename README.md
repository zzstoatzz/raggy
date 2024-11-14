# raggy

A Python library for scraping and document processing.

## Installation

```python
pip install raggy
```

For additional features:
```python
pip install raggy[scrapling]  # Enhanced web scraping via Scrapling
pip install raggy[chroma]     # ChromaDB support
pip install raggy[tpuf]       # TurboPuffer support
pip install raggy[pdf]        # PDF processing
```

Read the [docs](https://zzstoatzz.github.io/raggy/)

### What is it?

A Python library for:

- scraping the web to produce rich documents
- putting these documents in vectorstores
- querying the vectorstores to find documents similar to a query

> [!TIP]
> See this [example](https://github.com/zzstoatzz/raggy/blob/main/examples/chat_with_X/website.py) to chat with any website, or this [example](https://github.com/zzstoatzz/raggy/blob/main/examples/chat_with_X/repo.py) to chat with any GitHub repo.

### License and Dependencies

> [!IMPORTANT]
> This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
>
> When installing the optional `[scrapling]` dependency, please note that Scrapling is licensed under the BSD-3-Clause license. By using this optional feature, you agree to comply with [Scrapling's license terms](https://github.com/D4Vinci/Scrapling/blob/main/LICENSE).

### Contributing

We welcome contributions! See our [contributing guide](https://zzstoatzz.github.io/raggy/contributing) for details.
