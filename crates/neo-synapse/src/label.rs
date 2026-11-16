use serde::{Deserialize, Serialize};

/// Quality group of a cluster (phy's `cluster_group.tsv` values).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum UnitQualityLabel {
    /// Well-isolated single neuron (`"good"` in phy).
    #[serde(alias = "Good", alias = "good", alias = "singleunit")]
    SingleUnit,
    /// Multi-unit activity (`"mua"` in phy).
    #[serde(alias = "Mua", alias = "mua", alias = "multiunit")]
    MultiUnit,
    /// Artifact or noise (`"noise"` in phy).
    #[serde(alias = "noise")]
    Noise,
    /// Not curated (`"unsorted"` in phy).
    #[default]
    #[serde(alias = "unsorted")]
    Unsorted,
}

#[allow(non_upper_case_globals)]
impl UnitQualityLabel {
    /// Alias of phy's `"good"` group.
    pub const Good: Self = Self::SingleUnit;
    /// Alias of phy's `"mua"` group.
    pub const Mua: Self = Self::MultiUnit;

    pub fn as_str(self) -> &'static str {
        match self {
            Self::SingleUnit => "good",
            Self::MultiUnit => "mua",
            Self::Noise => "noise",
            Self::Unsorted => "unsorted",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "good" | "singleunit" | "sua" => Self::SingleUnit,
            "mua" | "multiunit" => Self::MultiUnit,
            "noise" => Self::Noise,
            _ => Self::Unsorted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip_through_their_phy_names() {
        for l in [
            UnitQualityLabel::Good,
            UnitQualityLabel::Mua,
            UnitQualityLabel::Noise,
            UnitQualityLabel::Unsorted,
        ] {
            assert_eq!(UnitQualityLabel::parse(l.as_str()), l);
        }
        assert_eq!(UnitQualityLabel::parse(" SUA "), UnitQualityLabel::Good);
        assert_eq!(
            UnitQualityLabel::parse("whatever"),
            UnitQualityLabel::Unsorted
        );
    }
}
