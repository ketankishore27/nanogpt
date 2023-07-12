from azureml.core import Workspace  

workspace = Workspace.create(name='mlexperiments',
                             subscription_id='9e0d323f-14e1-4270-a807-cab1ac20297a',
                             resource_group='mle-kk',
                             create_resource_group=False,
                             location='westeurope')

