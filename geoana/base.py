from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import traitlets as tr
from six import with_metaclass


class AutoDocumentor(type):

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
        traits_base = tuple([base._traits_class for base in bases])

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
        _traits_class = type(str('HasTraits'), (my_traits,) + traits_base, {})
        classdict["_traits_class"] = _traits_class

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
        newcls = super(AutoDocumentor, mcs).__new__(
            mcs, name, bases, classdict
        )
        return newcls


class BaseAnalytic(with_metaclass(AutoDocumentor)):

    def __init__(self, **kwargs):
        self.traits = self._traits_class()
        for key in kwargs:
            if key not in self.traits.trait_names():
                raise KeyError('{}: Keyword input is not trait'.format(key))
            setattr(self.traits, key, kwargs[key])
