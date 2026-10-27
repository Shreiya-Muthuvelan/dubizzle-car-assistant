import uuid
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Dubizzle Automotive Assistant",
    page_icon="🚗",
    layout="wide",
)

# -----------------------------
# Session state
# -----------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Helper functions
# -----------------------------
def start_new_conversation():
    """Reset the current UI conversation."""
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []

def format_value(value, prefix=""):
    """Format a listing value without displaying missing values."""
    if value is None or value == "":
        return None

    return f"{prefix}{value}"


def display_listing(listing):
    """Display user-friendly vehicle listing details."""

    make = listing.get("make") or "Unknown make"
    model = listing.get("model") or "Unknown model"

    st.markdown(f"**{make} {model}**")

    listing_id = listing.get("listing_id")
    if listing_id is not None:
        st.caption(f"Listing ID: {listing_id}")

    col1, col2, col3 = st.columns(3)

    with col1:
        price = listing.get("price_aed")
        if price is not None:
            st.metric("Selling price", f"AED {price:,.0f}")

    with col2:
        year = listing.get("year")
        if year is not None:
            st.metric("Year", str(year))

    with col3:
        mileage = listing.get("mileage_km")
        if mileage is not None:
            st.metric(
                "Mileage",
                f"{mileage:,.0f} km"
                if isinstance(mileage, (int, float))
                else str(mileage),
            )

    with st.expander("More vehicle details"):
        displayed = False

        trim = listing.get("trim")
        if trim:
            st.write(f"**Trim:** {trim}")
            displayed = True

        monthly_payment = listing.get("monthly_payment_aed")
        if monthly_payment is not None:
            st.write(f"**Monthly payment:** AED {monthly_payment:,.0f}")
            displayed = True

        description = listing.get("description")
        if description:
            st.write(f"**Description:** {description}")
            displayed = True

        photo_url = listing.get("photo_url")
        if photo_url:
            st.image(photo_url, width=300)
            displayed = True

        if not displayed:
            st.write("No additional details available.")

    st.divider()




def send_message(message):
    """Send a message to the existing FastAPI backend."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "conversation_id": st.session_state.conversation_id,
                "top_k": 5,
            },
            timeout=60,
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        st.error("The request took too long. Please try again.")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the assistant. "
            "Please make sure the FastAPI backend is running."
        )

    except requests.exceptions.HTTPError as exc:
        detail = "The backend could not process your request."

        try:
            detail = response.json().get("detail", detail)
        except (ValueError, AttributeError):
            pass

        st.error(f"{detail} (HTTP {response.status_code})")

    except requests.exceptions.RequestException:
        st.error("A network error occurred. Please try again.")

    except ValueError:
        st.error("The backend returned an invalid response.")

    return None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🚗 Automotive Assistant")
    st.write(
        "Find vehicles, compare listings, explore vehicle details, "
        "and arrange viewings."
    )
    st.button(
        "＋ New conversation",
        on_click=start_new_conversation,
        use_container_width=True,
    )

    st.divider()

    st.caption("Powered by FastAPI, semantic retrieval, and Gemini.")

    if st.button("Check backend status", use_container_width=True):
        try:
            health_response = requests.get(
                f"{API_URL}/health",
                timeout=5,
            )
            health_response.raise_for_status()
            st.success("Backend is reachable.")
        except requests.exceptions.RequestException:
            st.error("Backend is unavailable.")

# -----------------------------
# Main interface
# -----------------------------
st.title("Find your next car")
st.write(
    "Tell me what you're looking for, and I'll help you explore "
    "the available listings."
)

# Render existing conversation history.
for message_data in st.session_state.messages:
    with st.chat_message(message_data["role"]):
        st.markdown(message_data["content"])

        listings = message_data.get("results", [])
        if listings:
            st.caption(f"{len(listings)} vehicle(s) returned")
            for listing in listings:
                display_listing(listing)

        booking = message_data.get("booking")
        if booking:
            status = booking.get("status", "unknown").replace("_", " ").title()
            st.info(f"Viewing status: {status}")
            if booking.get("booking_id"):
                st.caption(f"Booking ID: {booking['booking_id']}")

        lead = message_data.get("lead")
        if lead:
            status = lead.get("status", "unknown").replace("_", " ").title()
            st.info(f"Lead qualification status: {status}")
            if lead.get("lead_id"):
                st.caption(f"Lead ID: {lead['lead_id']}")

# Chat input.
prompt = st.chat_input("Ask about cars, prices, mileage, or booking a viewing...")

if prompt:
    # Display and store the user's message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the message to FastAPI.
    with st.chat_message("assistant"):
        with st.spinner("Looking into that for you..."):
            data = send_message(prompt)

        if data is not None:
            answer = data.get(
                "answer",
                "I couldn't generate a response. Please try again.",
            )

            st.markdown(answer)

            results = data.get("results") or []
            if results:
                st.subheader("Matching vehicles")
                for listing in results:
                    display_listing(listing)

            booking = data.get("booking")
            if booking:
                status = booking.get("status", "unknown").replace("_", " ").title()
                st.info(f"Viewing status: {status}")
                if booking.get("booking_id"):
                    st.caption(f"Booking ID: {booking['booking_id']}")

            lead = data.get("lead")
            if lead:
                status = lead.get("status", "unknown").replace("_", " ").title()
                st.info(f"Lead qualification status: {status}")
                if lead.get("lead_id"):
                    st.caption(f"Lead ID: {lead['lead_id']}")

            st.session_state.conversation_id = data.get(
                "conversation_id",
                st.session_state.conversation_id,
            )

            # Save the assistant's response for subsequent reruns.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "results": results,
                    "booking": booking,
                    "lead": lead,
                }
            )