def load_targets(how='underscore', in_file='utils/targets.txt'):
    """Loads targets from input text file.
    
    Args:
        how: one of ['underscore', 'space', 'hyph', 'all'], outputting a list
            with underscored, space-separated, or hyphenated compounds, or all
            of those versions
        in_file: text file where first tab-separated column contains
            space-separated, two-constituent compounds.
    
    Returns:
        List of compounds.
    """
    
    with open(in_file, 'r') as f:
        targets = f.readlines()
    
    targets = [t.rstrip().split('\t')[0].lower() for t in targets]
    
    if how == 'underscore':
        targets = [t.replace(' ', '_') for t in targets]
    elif how == 'hyph':
        targets = [t.replace(' ', '-') for t in targets]
    elif how == 'all':
        targets = targets\
                + [t.replace(' ', '_') for t in targets]\
                + [t.replace(' ', '-') for t in targets]
        
    return targets