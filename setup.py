import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepmux",
    version="0.2.1",
    author="DeepMux",
    author_email="dev@deepmux.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Deep-Mux/deepmux-python",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'requests',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
