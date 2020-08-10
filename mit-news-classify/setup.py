import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mit-news-classify", # Replace with your own username
    version="0.8",
    author="Arun Wongprommoon",
    author_email="arunwpm@mit.edu",
    description="A news classification tool developed for Improve the News, a project by Max Tegmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.improvethenews.org/",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow>=2.1',
        'sklearn',
        'gensim',
        'transformers',
        'torch',
        'numpy',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)