"""Database initialization and seeding script."""
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal, init_db
from src.database.models import User, DatasetMeta
from src.auth.authentication import hash_password
from src.utils.config import settings
from src.utils.logger import logger


def seed_database(db: Session) -> None:
    """Seeds default admin and analyst users and initial dataset metadata."""
    # Ensure tables exist
    init_db()

    # Check if admin user exists
    admin = db.query(User).filter(User.username == settings.default_admin_user).first()
    if not admin:
        admin = User(
            username=settings.default_admin_user,
            email="admin@pricing-optimization.internal",
            hashed_password=hash_password(settings.default_admin_password),
            role="admin",
            is_active=True
        )
        db.add(admin)
        logger.info(f"Seeded default admin user: {settings.default_admin_user}")

    # Check if analyst user exists
    analyst = db.query(User).filter(User.username == settings.default_analyst_user).first()
    if not analyst:
        analyst = User(
            username=settings.default_analyst_user,
            email="analyst@pricing-optimization.internal",
            hashed_password=hash_password(settings.default_analyst_password),
            role="analyst",
            is_active=True
        )
        db.add(analyst)
        logger.info(f"Seeded default analyst user: {settings.default_analyst_user}")

    db.commit()


if __name__ == "__main__":
    db = SessionLocal()
    try:
        seed_database(db)
        print("Database successfully initialized and seeded.")
    finally:
        db.close()
