
def add_attributes(obj, locals: dict):
    for key, value in locals.items():
        if key != 'self' and key != '__class__': 
            setattr(obj, key, value)