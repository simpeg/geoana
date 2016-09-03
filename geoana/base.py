from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import traitlets as tr
from six import with_metaclass


class AutoDocumentor(tr.MetaHasTraits):

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

        def is_required(trait):
            return not trait.allow_none

        trait_dict = {}
        for base in reversed(bases):
            if issubclass(base, tr.HasTraits):
                trait_dict.update(base.class_traits())

        trait_dict.update(
            {
                key: value for key, value in classdict.items()
                if isinstance(value, tr.TraitType)
            }
        )

        doc_str = classdict.get('__doc__', '')
        trts = {key: value for key, value in trait_dict.items()}
        doc_str += '\n'.join(
            (value.sphinx(key) for key, value in trts.items())
        )
        classdict['__doc__'] = doc_str.strip()

        newcls = super(AutoDocumentor, mcs).__new__(
            mcs, name, bases, classdict
        )
        return newcls


class BaseAnalytic(with_metaclass(AutoDocumentor, tr.HasTraits)):

    def __init__(self, **kwargs):
        for key in kwargs:
            if key not in self.trait_names():
                raise KeyError('{}: Keyword input is not trait'.format(key))
        super(BaseAnalytic, self).__init__(**kwargs)
