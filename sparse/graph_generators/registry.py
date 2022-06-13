from sparse.graph_generators.generator import Erdos_Reyni

registered_generators =[Erdos_Reyni]

def get(gen_name):

    generator = None
    for registered_generator in registered_generators:
        if registered_generator.is_valid_name(gen_name):
            generator = registered_generator.get_generator_from_name(gen_name)
            break
    
    if generator is None:
        raise ValueError('No such generator: {}'.format(gen_name))
    
    return generator