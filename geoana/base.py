from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import traitlets as tr
from six import with_metaclass


class _PropertyMetaclass(type):

    def __new__(mcs, name, bases, classdict):

        def sphinx(trait_name, trait):
            if isinstance(trait, tr.TraitType) and hasattr(trait.sphinx):
                return trait.sphinx(trait_name)
            return (
                ':param {name}: {doc}\n:type {name}: '
                ':class:`{cls} <.{cls}>`'.format(
                    name=trait_name,
                    doc=trait.help,
                    cls=trait.__class__.__name__
                )
            )

        # Find all of the previous traits bases
        traits_base = tuple([base._backend_class for base in bases])

        # Grab all the traitlets stuff
        traits_dict = {
            key: value for key, value in classdict.items()
            if (
                isinstance(value, tr.TraitType) or
                isinstance(value, tr.ObserveHandler)
            )
        }

        # Create a new traits class and merge with previous
        my_traits = type(str('HasTraits'), (tr.HasTraits,), traits_dict)
        _backend_class = type(str('HasTraits'), (my_traits,) + traits_base, {})
        classdict["_backend_class"] = _backend_class

        # Overwrite the traits with properties, delete the others
        for n in traits_dict:
            trait = traits_dict[n]
            if isinstance(trait, tr.TraitType):
                classdict[n] = trait.get_property(n)
            else:
                del classdict[n]

        # Create some better documentation
        doc_str = classdict.get('__doc__', '')
        trts = {
            key: value for key, value in traits_dict.items()
            if isinstance(value, tr.TraitType)
        }
        doc_str += '\n'.join(
            (value.sphinx(key) for key, value in trts.items())
        )
        classdict["__doc__"] = __doc__

        # Create the new class
        newcls = super(_PropertyMetaclass, mcs).__new__(
            mcs, name, bases, classdict
        )
        return newcls


class HasProperties(with_metaclass(_PropertyMetaclass)):

    def __init__(self, **kwargs):
        self.backend = self._backend_class()
        for key in kwargs:
            if key not in self.backend.trait_names():
                raise KeyError('{}: Keyword input is not trait'.format(key))
            setattr(self.backend, key, kwargs[key])
