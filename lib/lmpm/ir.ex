defmodule IR.Value do
  defstruct [:id, :dtype, :shape, :clock, :storage, :partition]
end

defmodule IR.Visible do
  defstruct [:dst, :src, :pred, :attrs]
end

defmodule IR.Rule do
  defstruct [:id, :on, :reads, :backend, :emit]
end

defmodule IR.Constraint do
  defstruct [:id, :scope, :kind, :pred, :priority]
end

defmodule IR.Commit do
  defstruct [:target, :policy, :params]
end

defmodule IR.Index do
  defstruct [:epoch, :dst_ids, :src_ids, :segment_ptr, :attrs, :layout_hint]
end
