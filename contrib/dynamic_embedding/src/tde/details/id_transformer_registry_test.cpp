#include <iostream>
#include "gtest/gtest.h"
#include "tde/details/id_transformer_registry.h"
namespace tde::details {
TEST(TDE, IDTransformerRegistry) {
  ASSERT_NE(sizeof(TransformersVariant), 0);
}

} // namespace tde::details
