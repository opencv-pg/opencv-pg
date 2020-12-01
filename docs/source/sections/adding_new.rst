Adding New Transforms
=====================
In order to add new Transforms to the builtin list, the following steps must be completed:

#. Create a new Transform, subclassed from :class:`BaseTransform<opencv_pg.models.base_transform.BaseTransform>`.
#. Set the ``doc_filename`` class variable appropriately
    * Default doc filename will be the ``ClassName.html``
#. Create a new ``.html`` doc in ``src/opencv_pg/docs/source_docs/``. Follow the formats of the other files in the folder.
#. Add a new entry to the ``_TRANS_WINDOWS`` dict in ``src/opencv_pg/models/transform_windows.py`` with the new transform. This will add it to the Transforms list in the GUI.
