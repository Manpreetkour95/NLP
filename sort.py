# List of organization names
org_names = ['Alpin', 'RGH ltd', 'intel', 'huy co', 'Tech Centre', 'Health Service', 'ServiceNow', 'Centrelink']

# List of common suffixes to prioritize
suffixes = ['ltd', 'co']

# List of keywords to deprioritize
keywords = ['centre', 'service']

# Function to check if an organization name ends with any of the given suffixes
def has_common_suffix(name):
    for suffix in suffixes:
        if name.lower().endswith(suffix):
            return True
    return False

# Function to check if an organization name contains any of the given keywords
def contains_keyword(name):
    for keyword in keywords:
        if keyword in name.lower():
            return True
    return False
org_names =sorted(org_names , key=len, reverse=True)
print(org_names)
# Sort the organization names with custom key
sorted_org_names = sorted(
    org_names,
    key=lambda name: (not has_common_suffix(name), contains_keyword(name), name.lower())
)

print(sorted_org_names)
