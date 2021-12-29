import setuptools

setuptools.setup(
        name = 'xiebt',
        version = '0.1.0',
        packages = setuptools.find_packages(),
        install_requires=['fairseq'],
        entry_points = {
            'console_scripts':[
                'xiebt-generate = xiebt.generate:main']})

