from setuptools import setup, find_packages

setup(name='trainer',
        verison='1.0',
        packages=find_packages(),
        include_package_data=True,
        description='Training model for music generation using lstm',
        author='Yosua Muliawan',
        author_email='yosuamuliawan19@gmail.com',
        license='MIT',
        install_requires=[
            'keras==2.1.2', 
            'music21',
            'google-cloud-storage',
            'tensorflow-gpu',
            'h5py'
        ], 
        zip_safe=False)