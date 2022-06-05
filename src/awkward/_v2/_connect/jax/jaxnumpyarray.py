import awkward as ak


class JaxNumpyArray(ak._v2.contents.Content):
    nplike = ak.nplike.Jax.instance()

    def __init__(self, data, identifier=None, parameters=None):
        self.array = ak._v2.contents.NumpyArray(
            data, identifier, parameters, nplike=self.nplike
        )
        self._init(identifier, parameters, self.nplike)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def inner_shape(self):
        return self._data.shape[1:]

    @property
    def strides(self):
        return self._data.strides

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ptr(self):
        return self._data.device_buffer.unsafe_buffer_pointer()

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return JaxNumpyArray(
            self.raw(tt),
            self._typetracer_identifier(),
            self._parameters,
        )

    @property
    def length(self):
        return self._data.shape[0]

    def __getattr__(self, attr):
        return getattr(self.array, attr)
